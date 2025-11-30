r"""
ISCO Algorithm for MCut
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


import numpy as np
from typing import Callable
from ml4co_kit.task.graph.mcut import MCutTask
from ml4co_kit.solver.lib.isco.isco_base import (
    PASMHSampler, LinearTemperatureScheduler, EnergyFunction, metropolis_hastings_accept
)


class MCutEnergyFunction(EnergyFunction):
    def __init__(
        self, 
        edge_index: np.ndarray,
        edges_weight: np.ndarray,
        weights_matrix: np.ndarray
    ):
        # Super Initialization
        super(MCutEnergyFunction, self).__init__()

        # Set Attributes
        self.edge_index = edge_index
        self.edges_weight = edges_weight
        self.weights_matrix = weights_matrix
    
    def compute(self, x: np.ndarray) -> float:
        # Convert x_i in {0,1} to s_i in {-1, +1}
        s = 2 * x - 1
        
        # Energy: sum of s_i * s_j * w_ij for all edges
        # We want to maximize cut, so minimize this energy
        edge_i = self.edge_index[0]
        edge_j = self.edge_index[1]
        energy = float(np.sum(s[edge_i] * s[edge_j] * self.edges_weight))
        
        return energy
    
    def compute_delta_vector(self, x: np.ndarray) -> np.ndarray:
        # Convert to {-1, +1}
        s = 2 * x - 1
        
        # Gradient: 2 * (weights_matrix @ s)
        grad = 2 * (self.weights_matrix @ s)
        delta = (1 - 2 * x) * grad
        delta = np.clip(delta, -23, 23) # 1e-10 ~ 1e10
        return delta


def mcut_isco(
    task_data: MCutTask,
    isco_tau: float = 0.5, 
    isco_mu_init: float = 5.0,
    isco_g_func: Callable[[np.ndarray], np.ndarray] = lambda r: np.sqrt(r),
    isco_adapt_mu: bool = True,
    isco_target_accept_rate: float = 0.574,
    isco_iterations: int = 10000,
    isco_seed: int = 1234
):
    # Random seed
    np.random.seed(isco_seed)
    rng = np.random.RandomState(seed=isco_seed)
    
    # Preparation for decoding
    weights_matrix = task_data.to_adj_matrix(with_edge_weights=True)
    edge_index = task_data.edge_index
    edges_weight = task_data.edges_weight
    nodes_num: int = weights_matrix.shape[0]

    # Initial solutions
    x = np.random.randint(0, 2, size=nodes_num)
    
    # Temperature Scheduler
    temp_scheduler = LinearTemperatureScheduler(
        tau0=isco_tau, total_iterations=isco_iterations
    )

    # PASMHSampler
    sampler = PASMHSampler(
        rng=rng,
        mu_init=isco_mu_init,
        g_func=isco_g_func,
        adapt_mu=isco_adapt_mu,
        target_accept_rate=isco_target_accept_rate
    )

    # Energy Function
    energy_func = MCutEnergyFunction(edge_index, edges_weight, weights_matrix)

    # Initial energy and gradient
    best_sol = x.copy()
    best_energy = energy_func.compute(best_sol)
    current_energy = best_energy
    
    # ISCO
    for epoch in range(isco_iterations):
        # 1. Get current temperature
        tau = temp_scheduler.get_temperature(epoch)

        # 2. Generate proposal
        y, q_forward, q_reverse = sampler.propose(x, energy_func)
        
        # 3. Compute energy change
        proposed_energy = energy_func.compute(y)
        deltaE = proposed_energy - current_energy
        
        # 4. Accept/reject via MH criterion
        accepted = metropolis_hastings_accept(deltaE, tau, q_forward, q_reverse, rng)
        if accepted:
            x = y
            current_energy = proposed_energy
            
            # Update best solution if improved
            if current_energy < best_energy:
                best_energy = current_energy
                best_sol = x.copy()
        
        # 5. Adapt sampler parameters
        sampler.adapt(accepted)

    # Store the solution in the task_data
    task_data.from_data(sol=best_sol, ref=False)