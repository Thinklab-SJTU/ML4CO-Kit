r"""
Global Prediction Degree Solver.
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
from ml4co_kit.solver.lib.gp_degree.mcl_gp_degree import mcl_gp_degree_decoder
from ml4co_kit.solver.lib.gp_degree.mis_gp_degree import mis_gp_degree_decoder
from ml4co_kit.solver.lib.gp_degree.mvc_gp_degree import mvc_gp_degree_decoder


class GpDegreeSolver(SolverBase):
    def __init__(self, optimizer: OptimizerBase = None):
        super(GpDegreeSolver, self).__init__(
            solver_type=SOLVER_TYPE.GREEDY, optimizer=optimizer
        )

    def _solve(self, task_data: TaskBase):
        """Solve the task data using Global Prediction Degree Solver."""
        if task_data.task_type == TASK_TYPE.MCL:
            return mcl_gp_degree_decoder(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MIS:
            return mis_gp_degree_decoder(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_gp_degree_decoder(task_data=task_data)
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )