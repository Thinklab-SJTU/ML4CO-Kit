import torch
import numpy as np
from torch import Tensor
from typing import Tuple
from ml4co_kit.learning.utils import to_numpy, to_tensor


def mcut_rlsa_decoder(
    graph: np.ndarray,
    edge_index: np.ndarray,
    rlsa_tau: float = 1.5, 
    rlsa_d: int = 20, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_device: str = "cpu", 
    seed: int = 1234
) -> np.ndarray:
    # random seed
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    
    # preparation
    nodes_num = graph.shape[0]
    np.fill_diagonal(graph, 0)
    graph = to_tensor(graph).to(rlsa_device).float()
    edge_index = to_tensor(edge_index).to(rlsa_device).long()
    
    # initial solutions
    x = torch.randint(low=0, high=1, size=(rlsa_k, nodes_num))
    x = x.to(rlsa_device).float()
    x = torch.distributions.Bernoulli(x).sample().float()
    
    # initial energy and gradient
    energy, grad = mcut_energy_func(graph, edge_index, x)
    best_energy = energy.clone()
    best_sol = x.clone()
    
    # SA
    for epoch in range(rlsa_t):
        # temperature
        tau = rlsa_tau * (1 - epoch / rlsa_k)

        # sampling
        delta = grad * (2 * x - 1) / 2
        k = torch.randint(2, rlsa_d + 1, size=(1,)).item()
        term2 = -torch.kthvalue(-delta, k, dim=1, keepdim=True).values
        flip_prob = torch.sigmoid((delta - term2) / tau)
        rr = torch.rand_like(x.data.float())
        x = torch.where(rr < flip_prob, 1 - x, x)

        # update energy and gradient
        energy, grad = mcut_energy_func(graph, edge_index, x)
        to_update = energy < best_energy
        best_sol[to_update] = x[to_update]
        best_energy[to_update] = energy[to_update]
        
    # select the best
    edge_index_0 = 2 * best_sol[:, edge_index[0]] - 1
    edge_index_1 = 2 * best_sol[:, edge_index[1]] - 1
    edge_index_0_1 = edge_index_0 * edge_index_1
    sol_result = (edge_index_0_1 == -1).sum(dim=1)
    best_index = torch.argmax(sol_result, dim=0)
    return to_numpy(best_sol[best_index])


def mcut_energy_func(
    graph: Tensor, edge_index: Tensor, x: Tensor
) -> Tuple[Tensor, Tensor]:
    edge_index_0 = 2 * x[:, edge_index[0]] - 1
    edge_index_1 = 2 * x[:, edge_index[1]] - 1
    energy = torch.sum(edge_index_0 * edge_index_1, dim=1)
    grad = torch.matmul(2*x-1, graph)
    return energy, grad
