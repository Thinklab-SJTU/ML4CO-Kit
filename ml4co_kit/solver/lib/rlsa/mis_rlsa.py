r"""
RLSA Algorithm for MIS
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Union
from ml4co_kit.task.graph.mis import MISTask
from ml4co_kit.utils.type_utils import to_tensor, to_numpy
    

def mis_rlsa(
    task_data: MISTask,
    rlsa_init_type: str = "uniform",
    rlsa_kth_dim: Union[str, int] = 0,
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 5, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_alpha: float = 0.3,
    rlsa_beta: float = 1.02,
    rlsa_device: str = "cpu", 
    rlsa_seed: int = 1234
):
    # Random seed
    np.random.seed(seed=rlsa_seed)
    torch.manual_seed(seed=rlsa_seed)
    
    # Preparation for decoding
    adj_matrix = task_data.to_adj_matrix()
    nodes_weight = task_data.nodes_weight
    nodes_num = adj_matrix.shape[0]
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix = to_tensor(adj_matrix).to(rlsa_device).float()
    nodes_weight = to_tensor(nodes_weight).to(rlsa_device).float()

    # Initial solutions
    if rlsa_init_type == "gaussian":
        x = rlsa_alpha * torch.randn(size=(rlsa_k, nodes_num))
        x = torch.clip(x, 0, 1).to(rlsa_device).float()
    elif rlsa_init_type == "uniform":
        x = torch.randint(low=0, high=2, size=(rlsa_k, nodes_num))
        x = (rlsa_alpha * x).to(rlsa_device).float()
    else:
        raise NotImplementedError(
            "Only ``gaussian`` and ``uniform`` distributions are supported!"
        )
    x = torch.distributions.Bernoulli(x).sample().float()
    
    # Initial energy and gradient
    energy, grad = mis_energy_func(adj_matrix, x, nodes_weight, rlsa_beta)
    best_energy = energy.clone()
    best_sol = x.clone()
    
    # SA
    for epoch in range(rlsa_t):
        # kth_dim
        kth_dim = epoch % 2 if rlsa_kth_dim == "both" else rlsa_kth_dim
        
        # temperature
        tau = rlsa_tau * (1 - epoch / rlsa_k)

        # sampling
        delta = grad * (2 * x - 1) / 2
        k = torch.randint(2, rlsa_d + 1, size=(1,)).item()
        term2 = -torch.kthvalue(-delta, k, dim=kth_dim, keepdim=True).values
        flip_prob = torch.sigmoid((delta - term2) / tau)
        rr = torch.rand_like(x.data.float())
        x = torch.where(rr < flip_prob, 1 - x, x)

        # update energy and gradient
        energy, grad = mis_energy_func(adj_matrix, x, nodes_weight, rlsa_beta)
        to_update = energy < best_energy
        best_sol[to_update] = x[to_update]
        best_energy[to_update] = energy[to_update]
        
    # Select the best solution
    sol_uq = best_sol.unsqueeze(1)
    term2 = torch.sum((torch.matmul(sol_uq, adj_matrix) * sol_uq).squeeze(1), 1)
    meet_index = torch.where(term2 == 0)[0]
    best_index = meet_index[torch.argmax(best_sol[meet_index].sum(1))]

    # Store the solution in the task_data
    task_data.from_data(sol=to_numpy(best_sol[best_index]), ref=False)
    
    
def mis_energy_func(
    graph: Tensor, x: Tensor, weights: Tensor, penalty_coeff: float
) -> Tuple[Tensor, Tensor]:
    # Pre-Process
    x_uq = x.unsqueeze(1)
    
    # Energy Term1: Weighted Reward
    # Contribution of selecting node i: weight[i] * x[i]
    energy_term1 = torch.matmul(x, weights)
    
    # Energy Term2: Penalty for selecting nodes that are fully connected
    # If x_i = x_j = 1, while e_{i,j} = 1, incur penalty
    energy_term2 = torch.sum((torch.matmul(x_uq, graph) * x_uq).squeeze(1), 1)
    energy_term2 = penalty_coeff * energy_term2
    
    # Total Energy: maximize reward + minimize conflicts
    energy = -energy_term1 + energy_term2
    
    # Gradient Term 1: Weighted Reward
    grad_term1 = weights.expand_as(x)
    
    # Gradient Term 2: Penalty for selecting nodes that are fully connected
    grad_term2 = torch.matmul(graph, x.unsqueeze(-1)).squeeze(-1)
    grad_term2 = penalty_coeff * grad_term2
    
    # Total Gradient: maximize reward + minimize penalty 
    grad = -grad_term1 + grad_term2
    
    return energy, grad