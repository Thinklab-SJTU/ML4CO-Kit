r"""
Simulated Annealing Solver for Macro Placement.

References:
    [1] Mirhoseini et al., "A graph placement methodology for fast chip design",
        Nature, 2021.
    [2] Kirkpatrick et al., "Optimization by simulated annealing", Science, 1983.
    [3] Sechen & Sangiovanni-Vincentelli, "TimberWolf3.2: A new standard cell 
        placement and global routing package", DAC 1986.
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
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


def _sa_macro_placement(
    task_data: TaskBase,
    init_temp: float,
    cooling_rate: float,
    max_iter: int,
    perturbation_scale: float,
) -> TaskBase:
    """
    Simulated Annealing for Macro Placement.
    
    Each iteration perturbs a random macro's center position and accepts 
    the new solution according to the Metropolis criterion.
    """
    n = task_data.macros_num
    canvas_w = task_data.canvas_width
    canvas_h = task_data.canvas_height
    
    # Initialize solution: place macros at random valid positions
    sol = np.zeros((n, 2), dtype=task_data.precision)
    for i in range(n):
        w_i = task_data.macros[i]["width"]
        h_i = task_data.macros[i]["height"]
        sol[i, 0] = np.random.uniform(w_i / 2, canvas_w - w_i / 2)
        sol[i, 1] = np.random.uniform(h_i / 2, canvas_h - h_i / 2)
    
    current_cost = task_data.evaluate(sol)
    best_sol = sol.copy()
    best_cost = current_cost
    
    temp = init_temp
    for iteration in range(max_iter):
        # Select a random macro to perturb
        idx = np.random.randint(0, n)
        new_sol = sol.copy()
        
        dx = np.random.normal(0, perturbation_scale * canvas_w * temp / init_temp)
        dy = np.random.normal(0, perturbation_scale * canvas_h * temp / init_temp)
        new_sol[idx, 0] = np.clip(
            new_sol[idx, 0] + dx,
            task_data.macros[idx]["width"] / 2,
            canvas_w - task_data.macros[idx]["width"] / 2
        )
        new_sol[idx, 1] = np.clip(
            new_sol[idx, 1] + dy,
            task_data.macros[idx]["height"] / 2,
            canvas_h - task_data.macros[idx]["height"] / 2
        )
        
        new_cost = task_data.evaluate(new_sol)
        delta = new_cost - current_cost
        
        # Metropolis criterion
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            sol = new_sol
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_sol = sol.copy()
                best_cost = current_cost
        
        # Cool down
        temp *= cooling_rate
    
    task_data.sol = best_sol
    return task_data


class SimulatedAnnealingSolver(SolverBase):
    """
    Simulated Annealing Solver for EDA placement problems.
    
    Based on the classical SA approach used in early EDA placement tools 
    such as TimberWolf (Sechen & Sangiovanni-Vincentelli, DAC 1986).
    """
    def __init__(
        self,
        init_temp: float = 1000.0,
        cooling_rate: float = 0.995,
        max_iter: int = 50000,
        perturbation_scale: float = 0.05,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(SimulatedAnnealingSolver, self).__init__(
            solver_type=SOLVER_TYPE.SIMULATED_ANNEALING, optimizer=optimizer
        )
        
        # Initialize Attributes
        self.init_temp = init_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.perturbation_scale = perturbation_scale

    def _solve(self, task_data: TaskBase):
        if task_data.task_type == TASK_TYPE.MACRO_PLACEMENT:
            return _sa_macro_placement(
                task_data=task_data,
                init_temp=self.init_temp,
                cooling_rate=self.cooling_rate,
                max_iter=self.max_iter,
                perturbation_scale=self.perturbation_scale,
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )
