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
from typing import Tuple
from ml4co_kit.task.graph.mis import MISTask
from ml4co_kit.utils.type_utils import to_tensor, to_numpy
    

def mcut_rlsa(
    task_data: MISTask,
    rlsa_tau: float = 5, 
    rlsa_d: int = 20, 
    rlsa_k: int = 200, 
    rlsa_t: int = 200, 
    rlsa_device: str = "cpu",
    rlsa_dtype: torch.dtype = torch.float16, 
    rlsa_seed: int = 1234
):
    # Random seed
    np.random.seed(seed=rlsa_seed)
    torch.manual_seed(seed=rlsa_seed)
    
    # Process data for RLD
    nodes_num = task_data.nodes_num
    edge_index = to_tensor(task_data.edge_index).to(rlsa_device)
    edges_weight = to_tensor(task_data.edges_weight).to(rlsa_device, rlsa_dtype)
    A = torch.sparse_coo_tensor(
        edge_index, edges_weight, torch.Size((nodes_num, nodes_num))
    ).to_sparse_csr().to(rlsa_device, rlsa_dtype)
    b = edges_weight.reshape(-1, 1)

    # Initial solutions
    x = torch.randint(0,2, (nodes_num, rlsa_k), device=rlsa_device, dtype=rlsa_dtype)
    
    # Initial energy and gradient
    energy, grad = mcut_energy_func(A, x, edge_index, b, True)
    best_energy = energy.clone()
    best_sol = x.clone()
    
    # SA
    for epoch in range(rlsa_t):
        # Temperature
        tau = rlsa_tau * (1 - epoch / rlsa_t)

        # Sampling
        delta = grad * (2 * x - 1) / 2
        term2 = -torch.kthvalue(-delta, rlsa_d, dim=0, keepdim=True).values
        flip_prob = torch.sigmoid((delta - term2) / tau)
        rr = torch.rand_like(x.data)
        x = torch.where(rr < flip_prob, 1 - x, x)

        # Update energy and gradient
        energy, grad = mcut_energy_func(A, x, edge_index, b, True)
        best_sol = torch.where(
            (energy<best_energy).unsqueeze(0).repeat(nodes_num, 1), x, best_sol
        )
        best_energy = torch.where(energy<best_energy, energy, best_energy)
        
    # Decode
    best_sol = best_sol.transpose(0, 1)
    best_index = torch.argmin(best_energy)
    final_sol = best_sol[best_index]
    
    # Store the solution in the task_data
    task_data.from_data(sol=to_numpy(final_sol), ref=False)
    
    
def mcut_energy_func(
    A: Tensor, 
    x: Tensor, 
    edge_index: Tensor, 
    weights: Tensor,
    compute_grad: bool = False
) -> Tuple[Tensor, Tensor]:
    edge_index_0 = 2 * x[edge_index[0], :] - 1
    edge_index_1 = 2 * x[edge_index[1], :] - 1
    energy = torch.sum(edge_index_0 * edge_index_1 * weights, dim=0)
    grad = A @ (2*x-1) if compute_grad else None
    return energy, grad