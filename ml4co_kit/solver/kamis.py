r"""
KaMIS Solver.
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
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.kamis.mis_kamis import mis_kamis


class KaMISSolver(SolverBase):
    def __init__(
        self, 
        kamis_time_limit: float = 10.0,
        optimizer: OptimizerBase = None
    ):
        super(KaMISSolver, self).__init__(
            solver_type=SOLVER_TYPE.KAMIS, optimizer=optimizer
        )
        self.kamis_time_limit = kamis_time_limit

    def solve(self, task_data: TaskBase):
        """Solve the task data using KaMIS Solver."""
        if task_data.task_type == TASK_TYPE.MIS:
            return mis_kamis(
                task_data=task_data, 
                kamis_time_limit=self.kamis_time_limit
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )