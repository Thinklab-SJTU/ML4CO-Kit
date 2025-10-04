r"""
Local Construction Degree Solver.
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
from ml4co_kit.solver.lib.lc_degree.mcl_lc_degree import mcl_lc_degree
from ml4co_kit.solver.lib.lc_degree.mis_lc_degree import mis_lc_degree
from ml4co_kit.solver.lib.lc_degree.mvc_lc_degree import mvc_lc_degree
from ml4co_kit.solver.lib.lc_degree.mcut_lc_degree import mcut_lc_degree


class LcDegreeSolver(SolverBase):
    def __init__(self, optimizer: OptimizerBase = None):
        super(LcDegreeSolver, self).__init__(
            solver_type=SOLVER_TYPE.LC_DEGREE, optimizer=optimizer
        )

    def _solve(self, task_data: TaskBase):
        """Solve the task data using Local Construction Degree Solver."""
        if task_data.task_type == TASK_TYPE.MCL:
            return mcl_lc_degree(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MIS:
            return mis_lc_degree(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_lc_degree(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MCUT:
            return mcut_lc_degree(task_data=task_data)
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )