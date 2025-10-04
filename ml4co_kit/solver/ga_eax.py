r"""
GaEax Solver.
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


from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.ga_eax.tsp_ga_eax import tsp_ga_eax
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class GAEAXSolver(SolverBase):
    def __init__(
        self,
        ga_eax_scale: int = 1e5,
        ga_eax_max_trials: int = 1,
        ga_eax_population_num: int = 100,
        ga_eax_offspring_num: int = 30,
        ga_eax_show_info: bool = False,
        use_large_solver: bool = False,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(GAEAXSolver, self).__init__(SOLVER_TYPE.GA_EAX, optimizer=optimizer)

        # Initialize Attributes
        self.ga_eax_scale = ga_eax_scale
        self.ga_eax_max_trials = ga_eax_max_trials
        self.ga_eax_population_num = ga_eax_population_num
        self.ga_eax_offspring_num = ga_eax_offspring_num
        self.ga_eax_show_info = ga_eax_show_info
        self.use_large_solver = use_large_solver
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using GaEax solver."""
        if task_data.task_type == TASK_TYPE.TSP:
            return tsp_ga_eax(
                task_data=task_data,
                ga_eax_population_num=self.ga_eax_population_num,
                ga_eax_offspring_num=self.ga_eax_offspring_num,
                ga_eax_show_info=self.ga_eax_show_info,
                use_large_solver=self.use_large_solver
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )