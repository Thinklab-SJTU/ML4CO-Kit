r"""
FEM Solver for Minimum Vertex Cover (MVC) Problem.
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
from ml4co_kit.task.graph.mvc import MVCTask


def energy_mvc(adj: Tensor, p: Tensor, penalty: float = 10.0) -> Tensor:
    """
    Energy function for MVC with penalty for uncovered edges.
    
    E = Σ_i p_i + penalty * Σ_{(i,j)∈E} (1 - p_i) * (1 - p_j)
    
    Args:
        adj: Adjacency matrix [N, N]
        p: Probability matrix [batch, N]
        penalty: Penalty weight for uncovered edges
        
    Returns:
        Energy values [batch]
    """
    # Minimize number of selected nodes
    cost = p.sum(1)
    
    # Penalize uncovered edges
    uncovered = ((1 - p) @ adj) * (1 - p)
    constraint_violation = uncovered.sum(1) / 2  # Divide by 2 for undirected graph
    
    return cost + penalty * constraint_violation


def entropy_mvc(p: Tensor) -> Tensor:
    """Binary entropy for MVC."""
    return - ((p * torch.log(p + 1e-10)) + (1 - p) * torch.log(1 - p + 1e-10)).sum(1)


def mvc_fem(
    task_data: MVCTask,
    num_trials: int,
    betas: Tensor,
    grad_opt_class: Callable,
    device: str = "cpu",
    seed: int = 1,
    h_factor: float = 0.01,
    penalty: float = 10.0,
):
    """
    Solve MVC problem using FEM algorithm.
    
    Args:
        task_data: MVCTask instance
        num_trials: Number of parallel trials
        betas: Inverse temperature schedule
        grad_opt_class: Gradient optimizer class
        device: 'cuda' or 'cpu'
        seed: Random seed
        h_factor: Initial field strength
        penalty: Penalty for uncovered edges
    """
    # Set random seed
    torch.manual_seed(seed)

    # Extract graph information
    nodes_num = task_data.nodes_num
    adj_matrix = task_data.to_adj_matrix(with_edge_weights=False)
    adj_matrix = to_tensor(adj_matrix).float().to(device)

    # Initialize random fields h (bias towards selecting nodes)
    h = h_factor * torch.randn([num_trials, nodes_num]).float().to(device) + 1.0
    h.requires_grad = True

    # Setup optimizer
    grad_opt: torch.optim.Optimizer = grad_opt_class([h])

    # Main annealing loop
    for beta in betas:
        probs = torch.sigmoid(h)
        grad_opt.zero_grad()
        energy = energy_mvc(adj_matrix, probs, penalty)
        entropy = entropy_mvc(probs)
        free_energy = energy - entropy / beta
        free_energy.backward(gradient=torch.ones_like(free_energy))
        grad_opt.step()

    # Decoding: select nodes with probability > 0.5
    sols = (probs > 0.5).float()
    
    # Post-processing: ensure all edges are covered (greedy repair)
    sols = _repair_coverage(sols, adj_matrix)

    # Get the best solution (minimize size)
    sizes = sols.sum(1)
    min_idx = torch.argmin(sizes)
    best_sol = to_numpy(sols[min_idx])

    # Store the solution in the task_data
    task_data.from_data(sol=best_sol, ref=False)


def _repair_coverage(sols: Tensor, adj: Tensor) -> Tensor:
    """
    Repair solutions to ensure all edges are covered.
    
    For each solution, if an edge is uncovered, add one of its endpoints.
    """
    batch_size, n = sols.shape
    
    for i in range(batch_size):
        sol = sols[i]
        
        # Find uncovered edges
        for u in range(n):
            if sol[u] < 0.5:  # u is not in cover
                neighbors = torch.where(adj[u] > 0)[0]
                for v in neighbors:
                    if sol[v] < 0.5:  # v is also not in cover, edge (u,v) is uncovered
                        # Add the node with higher degree
                        deg_u = adj[u].sum()
                        deg_v = adj[v].sum()
                        if deg_u >= deg_v:
                            sol[u] = 1
                        else:
                            sol[v] = 1
                        break
    
    return sols
