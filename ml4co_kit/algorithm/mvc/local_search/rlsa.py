import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Union
from ml4co_kit.learning.utils import to_numpy, to_tensor


def mvc_rlsa_local_search(
    init_sol: np.ndarray,
    graph: np.ndarray,
    rlsa_init_type: str = "uniform",
    rlsa_kth_dim: Union[str, int] = "both",
    rlsa_tau: float = 0.01, 
    rlsa_d: int = 5, 
    rlsa_k: int = 1000, 
    rlsa_t: int = 1000, 
    rlsa_alpha: float = 0.3,
    rlsa_beta: float = 1.02,
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
    
    # initial solutions
    x = init_sol.repeat(rlsa_k, 1).to(rlsa_device).float()
    if rlsa_init_type == "gaussian":
        x[1:] = rlsa_alpha * torch.randn_like(x[1:])
        x = torch.clip(x, 0, 1).float()
        x[1:] = torch.ones_like(x[1:]) - x[1:]
    elif rlsa_init_type == "uniform":
        x[1:] = rlsa_alpha * torch.randint_like(x[1:], high=2)
        x[1:] = torch.ones_like(x[1:]) - x[1:]
    else:
        raise NotImplementedError(
            "Only ``gaussian`` and ``uniform`` distributions are supported!"
        )
    x = torch.distributions.Bernoulli(x).sample().float()
    
    # initial energy and gradient
    energy, grad = mvc_energy_func(graph, x, rlsa_beta)
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
        energy, grad = mvc_energy_func(graph, x, rlsa_beta)
        to_update = energy < best_energy
        best_sol[to_update] = x[to_update]
        best_energy[to_update] = energy[to_update]
        
    # select the best
    minus_sol = torch.ones_like(best_sol) - best_sol
    minus_sol_uq = minus_sol.unsqueeze(1)
    term2 = torch.sum((torch.matmul(minus_sol_uq, graph) * minus_sol_uq).squeeze(1), 1)
    meet_index = torch.where(term2 == 0)[0]
    best_index = meet_index[torch.argmin(best_sol[meet_index].sum(1))]
    return to_numpy(best_sol[best_index])


def mvc_energy_func(
    graph, x: Tensor, penalty_coeff: float
) -> Tuple[Tensor, Tensor]:
    minus_x = torch.ones_like(x) - x
    minus_x_uq = minus_x.unsqueeze(1)
    energy_term1 = torch.sum(x, dim=1)
    energy_term2 = torch.sum((torch.matmul(minus_x_uq, graph) * minus_x_uq).squeeze(1), 1)
    energy = energy_term1 + penalty_coeff * energy_term2
    grad_term1 = torch.ones_like(x)
    grad_term2 = penalty_coeff * torch.matmul(graph, minus_x.unsqueeze(-1)).squeeze(-1)
    grad = grad_term1 - grad_term2
    return energy, grad