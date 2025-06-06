r"""
OR-Tools for solving KP.

OR-Tools is open source software for combinatorial optimization, which seeks to 
find the best solution to a problem out of a very large set of possible solutions.

We follow https://developers.google.cn/optimization.
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


import copy
import numpy as np
from typing import Union
from multiprocessing import Pool
from ortools.algorithms.python import knapsack_solver
from ml4co_kit.solver.kp.base import KPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class KPORSolver(KPSolver):
    def __init__(self, time_limit: float = 60.0):
        super(KPORSolver, self).__init__(
            solver_type=SOLVER_TYPE.ORTOOLS, time_limit=time_limit
        )
        
    def solve(
        self,
        weights: Union[list, np.ndarray] = None,
        values: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
        num_threads: int = 1,
        apply_scale: bool = True,
        show_time: bool = False,
    ) -> np.ndarray:
        # preparation
        if weights is not None:
            self.from_data(
                weights=weights, values=values, capacities=capacities
            )
        if apply_scale:
            ori_weights, ori_values, ori_capacities = self.weights, self.values, self.capacities
            self.weights, self.values, self.capacities = self._apply_scale_and_dtype(
                weights=self.weights, values=self.values, capacities=self.capacities
            )
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        solutions = list()
        instance_num = len(self.weights)
        if num_threads == 1:
            for idx in iterative_execution(range, instance_num, self.solve_msg, show_time):
                solutions.append(self._solve(idx=idx))
        else:
            for idx in iterative_execution(
                range, instance_num // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_sols = p1.map(
                        self._solve,
                        [
                            idx * num_threads + inner_idx
                            for inner_idx in range(num_threads)
                        ],
                    )
                for sol in cur_sols:
                    solutions.append(sol)
            
        # restore solutions
        self.from_data(items_label=solutions, ref=False)
        
        # show time
        timer.end()
        timer.show_time()

        if apply_scale:
            self.weights, self.values, self.capacities = ori_weights, ori_values, ori_capacities
        
        return self.items_label
    
    def _solve(self, idx: int) -> np.ndarray:
        # init solver
        or_solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",
        )
        or_solver.set_time_limit(self.time_limit)
        weights = self.weights[idx]
        values = self.values[idx]
        capacity = self.capacities[idx]
        or_solver.init(values, [weights], [capacity])

        # solve
        or_solver.solve()

        # obtain solution
        items_num = len(values)
        sol = np.zeros(shape=(items_num,))
        sel_items_idx = [i for i in range(items_num) if or_solver.best_solution_contains(i)]
        sol[sel_items_idx] = 1

        return sol
    
    def __str__(self) -> str:
        return "KPORSolver"
