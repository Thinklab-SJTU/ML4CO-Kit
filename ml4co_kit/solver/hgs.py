r"""
HGS Solver.
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
from ml4co_kit.solver.lib.hgs.cvrp_hgs import cvrp_hgs
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class HGSSolver(SolverBase):
    def __init__(
        self, 
        hgs_scale: int = 2e4,
        hgs_demands_scale: int = 1e5,
        hgs_time_limit: float = 1.0,
        hgs_show_info: bool = False,
        optimizer: OptimizerBase = None
    ):
        super(HGSSolver, self).__init__(
            solver_type=SOLVER_TYPE.HGS, optimizer=optimizer
        )
        
        # Set Attributes
        self.hgs_scale = hgs_scale
        self.hgs_demands_scale = hgs_demands_scale
        self.hgs_time_limit = hgs_time_limit
        self.hgs_show_info = hgs_show_info
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using HGS solver."""
        if task_data.task_type == TASK_TYPE.CVRP:
            return cvrp_hgs(
                task_data=task_data,
                hgs_scale=self.hgs_scale,
                hgs_demands_scale=self.hgs_demands_scale,
                hgs_time_limit=self.hgs_time_limit,
                hgs_show_info=self.hgs_show_info
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )