import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Union
from ml4co_kit.learning.utils import to_numpy, to_tensor


def mcut_rlsa_local_search(
    init_sol: np.ndarray,
    graph: np.ndarray,
    edge_index: np.ndarray,
    rlsa_kth_dim: Union[str, int] = 0,
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 2, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_device: str = "cpu", 
    seed: int = 1234
) -> np.ndarray:
    # random seed
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    
    # preparation
    init_sol: Tensor = to_tensor(init_sol)
    np.fill_diagonal(graph, 0)
    graph = to_tensor(graph).to(rlsa_device).float()
    edge_index = to_tensor(edge_index).to(rlsa_device).long()
    
    # initial solutions
    x = init_sol.repeat(rlsa_k, 1).to(rlsa_device).float()
    x[1:] = torch.randint_like(x[1:], high=2).float()
    x = torch.distributions.Bernoulli(x).sample().float()
    
    # initial energy and gradient
    energy, grad = mcut_energy_func(graph, edge_index, x)
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
