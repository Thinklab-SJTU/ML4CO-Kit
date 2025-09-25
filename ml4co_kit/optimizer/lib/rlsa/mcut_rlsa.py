r"""
RLSA local search algorithm for MCl.
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
import numpy as np
from torch import Tensor
from typing import Tuple, Union
from ml4co_kit.task.graph.mcut import MCutTask
from ml4co_kit.utils.type_utils import to_numpy, to_tensor


def mcut_rlsa_ls(
    task_data: MCutTask,
    rlsa_kth_dim: Union[str, int] = "both",
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 2, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_device: str = "cpu", 
    rlsa_seed: int = 1234
):
    """RLSA local search for MCl problems."""
    # Random seed
    np.random.seed(seed=rlsa_seed)
    torch.manual_seed(seed=rlsa_seed)

    # Preparation for local search
    weights_matrix = task_data.to_adj_matrix(with_edge_weights=True)
    adj_matrix = task_data.to_adj_matrix()
    edge_index = task_data.edge_index
    edges_weight = task_data.edges_weight
    adj_matrix = to_tensor(adj_matrix).to(rlsa_device).float()
    edge_index = to_tensor(edge_index).to(rlsa_device).long()
    edges_weight = to_tensor(edges_weight).to(rlsa_device).float()
    weights_matrix = to_tensor(weights_matrix).to(rlsa_device).float()
    
    # Initial solutions
    init_sol: Tensor = to_tensor(task_data.sol)
    x = init_sol.repeat(rlsa_k, 1).to(rlsa_device).float()
    x[1:] = torch.randint_like(x[1:], high=2).float()
    x = torch.distributions.Bernoulli(x).sample().float() 
    
    # Initial energy and gradient
    energy, grad = mcut_energy_func(
        edge_index, x, edges_weight, weights_matrix
    )
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
        energy, grad = mcut_energy_func(
            edge_index, x, edges_weight, weights_matrix
        )
        to_update = energy < best_energy
        best_sol[to_update] = x[to_update]
        best_energy[to_update] = energy[to_update]
        
    # Select the best solution
    best_index = torch.argmax(best_energy)

    # Store the solution in the task_data
    task_data.from_data(sol=to_numpy(best_sol[best_index]), ref=False)
    
    
def mcut_energy_func(
    edge_index: Tensor, x: Tensor, weights: Tensor, weights_graph: Tensor
) -> Tuple[Tensor, Tensor]:
    # x_i in {0,1} -> s_i in {-1, +1}
    edge_index_0 = 2 * x[:, edge_index[0]] - 1
    edge_index_1 = 2 * x[:, edge_index[1]] - 1
    
    # Energy
    energy = torch.sum(edge_index_0 * edge_index_1 * weights, dim=1)
    
    # Gradient
    grad = torch.matmul(2*x-1, weights_graph)
    return energy, grad