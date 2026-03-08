r"""
FEM Solver for Max Cut Problem.

This module implements the Free Energy Minimization (FEM) algorithm for solving
the Maximum Cut problem using mean-field approximation and simulated annealing.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import torch
from torch import Tensor
from typing import Callable
from ml4co_kit.utils import to_tensor, to_numpy
from ml4co_kit.task.graph.mcut import MCutTask


def energy_mcut(J: Tensor, p: Tensor) -> Tensor:
    """
    p is the marginal matrix, with shape [batch, N]
    config is the configuration for n variables, with shape [batch, N]
    return TWICE the expected cut size, i.e. outer weights. 
    """
    return 2 * ((p @ J) * (1-p)).sum(1)


def entropy_mcut(p: Tensor) -> Tensor:
    return - ((p*torch.log(p)) + (1-p)*torch.log(1-p)).sum(1)


def mcut_fem(
    task_data: MCutTask,
    num_trials: int,
    betas: Tensor,
    grad_opt_class: Callable,
    device: str = "cpu",
    seed: int = 1,
    h_factor: float = 0.01,
):
    """
    Solve Max Cut problem using FEM algorithm.
    """ 
    # Set random seed
    torch.manual_seed(seed)

    # Extract graph information
    nodes_num = task_data.nodes_num
    adj_matrix = task_data.to_adj_matrix(
        with_edge_weights=task_data.edge_weighted
    )
    adj_matrix = to_tensor(adj_matrix).float().to(device)

    # Initialize random fields h
    h = h_factor * torch.randn([num_trials, nodes_num]).float().to(device)
    h.requires_grad = True

    # Setup optimizer
    grad_opt: torch.optim.Optimizer = grad_opt_class([h])

    # Main annealing loop
    for beta in betas:
        probs = torch.sigmoid(h)
        grad_opt.zero_grad()
        energy = -energy_mcut(adj_matrix / 2, probs)
        entropy = entropy_mcut(probs)
        free_energy = energy - entropy / beta
        free_energy.backward(gradient=torch.ones_like(free_energy))
        grad_opt.step()

    # Decoding probs
    sols = (probs > 0.5).float()

    # Get the best solution
    costs = energy_mcut(adj_matrix, sols) / 2
    max_idx = torch.argmax(costs)
    best_sol = to_numpy(sols[max_idx])

    # Store the solution in the task_data
    task_data.from_data(sol=best_sol, ref=False)