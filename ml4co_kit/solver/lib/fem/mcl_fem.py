r"""
FEM Solver for Maximum Clique (MCL) Problem.
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
from ml4co_kit.task.graph.mcl import MClTask
from ml4co_kit.utils import to_tensor, to_numpy


def energy_mcl(adj: Tensor, p: Tensor, penalty: float = 10.0) -> Tensor:
    """
    Energy function for MCL with penalty for non-adjacent selected nodes.
    
    E = -Σ_i p_i + penalty * Σ_{i,j: (i,j)∉E} p_i * p_j
    
    Args:
        adj: Adjacency matrix [N, N]
        p: Probability matrix [batch, N]
        penalty: Penalty weight for constraint violation
        
    Returns:
        Energy values [batch]
    """
    # Maximize number of selected nodes (negative sign)
    reward = -p.sum(1)
    
    # Penalize non-adjacent nodes being selected together
    # For clique, all selected nodes must be pairwise adjacent
    # Violation = Σ_{i,j} p_i * p_j * (1 - adj_{ij})
    n = adj.shape[0]
    complement_adj = 1 - adj - torch.eye(n, device=adj.device)  # Complement graph
    constraint_violation = ((p @ complement_adj) * p).sum(1) / 2
    
    return reward + penalty * constraint_violation


def entropy_mcl(p: Tensor) -> Tensor:
    """Binary entropy for MCL."""
    return - ((p * torch.log(p + 1e-10)) + (1 - p) * torch.log(1 - p + 1e-10)).sum(1)


def mcl_fem(
    task_data: MClTask,
    num_trials: int,
    betas: Tensor,
    grad_opt_class: Callable,
    device: str = "cpu",
    seed: int = 1,
    h_factor: float = 0.01,
    penalty: float = 10.0,
):
    """
    Solve MCL problem using FEM algorithm.
    
    Args:
        task_data: MCLTask instance
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
        energy = energy_mcl(adj_matrix, probs, penalty)
        entropy = entropy_mcl(probs)
        free_energy = energy - entropy / beta
        free_energy.backward(gradient=torch.ones_like(free_energy))
        grad_opt.step()

    # Decoding: select nodes with probability > 0.5
    sols = (probs > 0.5).float()
    
    # Post-processing: ensure clique property (greedy repair)
    sols = _repair_clique(sols, adj_matrix)

    # Get the best solution (maximize size)
    sizes = sols.sum(1)
    max_idx = torch.argmax(sizes)
    best_sol = to_numpy(sols[max_idx])

    # Store the solution in the task_data
    task_data.from_data(sol=best_sol, ref=False)


def _repair_clique(sols: Tensor, adj: Tensor) -> Tensor:
    """
    Repair solutions to ensure clique property.
    
    For each solution, remove nodes that are not connected to all other selected nodes.
    """
    batch_size, n = sols.shape
    
    for i in range(batch_size):
        sol = sols[i]
        selected = torch.where(sol > 0.5)[0]
        
        if len(selected) <= 1:
            continue
        
        # Check clique property
        changed = True
        while changed:
            changed = False
            for node in selected:
                if sol[node] < 0.5:
                    continue
                    
                # Check if node is connected to all other selected nodes
                other_selected = selected[selected != node]
                if len(other_selected) == 0:
                    continue
                    
                # Check connectivity
                connected_to_all = True
                for other in other_selected:
                    if sol[other] > 0.5 and adj[node, other] < 0.5:
                        connected_to_all = False
                        break
                
                if not connected_to_all:
                    sol[node] = 0
                    changed = True
            
            selected = torch.where(sol > 0.5)[0]
    
    return sols
