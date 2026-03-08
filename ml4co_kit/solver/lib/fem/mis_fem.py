r"""
FEM Solver for Maximum Independent Set (MIS) Problem.
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
from ml4co_kit.task.graph.mis import MISTask


def energy_mis(adj: Tensor, p: Tensor, penalty: float = 10.0) -> Tensor:
    """
    Energy function for MIS with penalty for adjacent selected nodes.
    
    E = -Σ_i p_i + penalty * Σ_{(i,j)∈E} p_i * p_j
    
    Args:
        adj: Adjacency matrix [N, N]
        p: Probability matrix [batch, N]
        penalty: Penalty weight for constraint violation
        
    Returns:
        Energy values [batch]
    """
    # Maximize number of selected nodes (negative sign)
    reward = -p.sum(1)
    
    # Penalize adjacent nodes being selected together
    constraint_violation = ((p @ adj) * p).sum(1)
    
    return reward + penalty * constraint_violation


def entropy_mis(p: Tensor) -> Tensor:
    """Binary entropy for MIS."""
    return - ((p * torch.log(p + 1e-10)) + (1 - p) * torch.log(1 - p + 1e-10)).sum(1)


def mis_fem(
    task_data: MISTask,
    num_trials: int,
    betas: Tensor,
    grad_opt_class: Callable,
    device: str = "cpu",
    seed: int = 1,
    h_factor: float = 0.01,
    penalty: float = 10.0,
):
    """
    Solve MIS problem using FEM algorithm.
    
    Args:
        task_data: MISTask instance
        num_trials: Number of parallel trials
        betas: Inverse temperature schedule
        grad_opt_class: Gradient optimizer class
        device: 'cuda' or 'cpu'
        seed: Random seed
        h_factor: Initial field strength
        penalty: Penalty for constraint violation
    """
    # Set random seed
    torch.manual_seed(seed)

    # Extract graph information
    nodes_num = task_data.nodes_num
    adj_matrix = task_data.to_adj_matrix(with_edge_weights=False)
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
        energy = energy_mis(adj_matrix, probs, penalty)
        entropy = entropy_mis(probs)
        free_energy = energy - entropy / beta
        free_energy.backward(gradient=torch.ones_like(free_energy))
        grad_opt.step()

    # Decoding: select nodes with probability > 0.5
    sols = (probs > 0.5).float()
    
    # Post-processing: ensure independence (greedy repair)
    sols = _repair_independence(sols, adj_matrix)

    # Get the best solution (maximize size)
    sizes = sols.sum(1)
    max_idx = torch.argmax(sizes)
    best_sol = to_numpy(sols[max_idx])

    # Store the solution in the task_data
    task_data.from_data(sol=best_sol, ref=False)


def _repair_independence(sols: Tensor, adj: Tensor) -> Tensor:
    """
    Repair solutions to ensure independence constraint.
    
    For each solution, if two adjacent nodes are both selected,
    randomly keep one and remove the other.
    """
    batch_size, n = sols.shape
    
    for i in range(batch_size):
        sol = sols[i]
        selected = torch.where(sol > 0.5)[0]
        
        # Check for violations
        for node in selected:
            neighbors = torch.where(adj[node] > 0)[0]
            selected_neighbors = neighbors[sol[neighbors] > 0.5]
            
            if len(selected_neighbors) > 0:
                # Keep current node, remove neighbors
                sol[selected_neighbors] = 0
    
    return sols
