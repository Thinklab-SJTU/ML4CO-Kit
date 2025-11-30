r"""
ISCO Algorithm for MCl
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
from ml4co_kit.task.graph.mcl import MClTask
from ml4co_kit.solver.lib.isco.isco_base import (
    PASMHSampler, LinearTemperatureScheduler, EnergyFunction, metropolis_hastings_accept
)


class MClEnergyFunction(EnergyFunction):
    def __init__(
        self, 
        adj_matrix: np.ndarray, 
        nodes_weight: np.ndarray, 
        penalty_coeff: float
    ):
        # Super Initialization
        super(MClEnergyFunction, self).__init__()

        # Set Attributes
        self.adj_matrix = adj_matrix
        self.nodes_weight = nodes_weight
        self.penalty_coeff = penalty_coeff
    
    def compute(self, x: np.ndarray) -> float:
        # Compute complement adjacency matrix (edges that don't exist)
        comp_adj_matrix = 1 - self.adj_matrix
        comp_Ax = comp_adj_matrix @ x
        
        # Objective: maximize weighted nodes (negative for minimization)
        objective = -float(self.nodes_weight.dot(x))
        
        # Penalty: penalize selecting nodes that are not fully connected
        # If x_i = x_j = 1 but edge (i,j) doesn't exist, incur penalty
        penalty = 0.5 * self.penalty_coeff * float(x @ comp_Ax)
        
        energy = objective + penalty
        return energy
    
    def compute_delta_vector(self, x: np.ndarray) -> np.ndarray:
        # Compute complement adjacency matrix
        comp_adj_matrix = 1 - self.adj_matrix
        comp_Ax = comp_adj_matrix @ x
        
        # Gradient: -weight + penalty * (complement adjacency @ x)
        grad = -self.nodes_weight + self.penalty_coeff * comp_Ax
        delta = (1 - 2 * x) * grad
        delta = np.clip(delta, -23, 23) # 1e-10 ~ 1e10
        return delta


def mcl_isco(
    task_data: MClTask,
    isco_init_type: str = "uniform",
    isco_tau: float = 0.5, 
    isco_mu_init: float = 5.0,
    isco_g_func: Callable[[np.ndarray], np.ndarray] = lambda r: np.sqrt(r),
    isco_adapt_mu: bool = True,
    isco_target_accept_rate: float = 0.574,
    isco_alpha: float = 0.3,
    isco_beta: float = 1.002,
    isco_iterations: int = 10000,
    isco_seed: int = 1234
):
    # Random seed
    np.random.seed(isco_seed)
    rng = np.random.RandomState(seed=isco_seed)
    
    # Preparation for decoding
    adj_matrix = task_data.to_adj_matrix()
    nodes_weight = task_data.nodes_weight
    nodes_num = adj_matrix.shape[0]
    np.fill_diagonal(adj_matrix, 1)  # For MCl, diagonal should be 1 (node connects to itself)

    # Initial solutions
    if isco_init_type == "gaussian":
        probs = isco_alpha * np.random.randn(nodes_num)
        probs = np.clip(probs, 0, 1).astype(np.float32)
    elif isco_init_type == "uniform":
        probs: np.ndarray = np.random.randint(0, 2, size=nodes_num)
        probs = (isco_alpha * probs).astype(np.float32)
    else:
        raise NotImplementedError(
            "Only ``gaussian`` and ``uniform`` distributions are supported!"
        )
    x = (rng.rand(nodes_num) < probs).astype(np.float32)
    
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
    energy_func = MClEnergyFunction(adj_matrix, nodes_weight, isco_beta)

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