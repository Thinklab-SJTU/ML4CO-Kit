r"""
Pygmtools Solver.
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


from typing import List
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.pygm.gm_pygm import gm_pygm, gm_pygm_batch
from ml4co_kit.solver.lib.pygm.ged_pygm import ged_pygm, ged_pygm_batch
from ml4co_kit.solver.lib.pygm.kqap_pygm import kqap_pygm, kqap_pygm_batch
from ml4co_kit.extension.pygmtools import PyGMToolsQAPSolver


class PyGMSolver(SolverBase):
    def __init__(
        self,
        pygm_qap_solver: PyGMToolsQAPSolver = PyGMToolsQAPSolver(),
        optimizer: OptimizerBase = None
    ):
        super(PyGMSolver, self).__init__(
            solver_type=SOLVER_TYPE.PYGM, optimizer=optimizer
        )

        # Initialize Attributes
        self.pygm_qap_solver = pygm_qap_solver
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using Pygmtools solver."""
        if task_data.task_type == TASK_TYPE.GM:
            return gm_pygm(
                task_data=task_data, pygm_qap_solver=self.pygm_qap_solver
            )
        elif task_data.task_type == TASK_TYPE.GED:
            return ged_pygm(
                task_data=task_data, pygm_qap_solver=self.pygm_qap_solver
            )
        elif task_data.task_type == TASK_TYPE.KQAP:
            return kqap_pygm(
                task_data=task_data, pygm_qap_solver=self.pygm_qap_solver
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )

    def _batch_solve(self, batch_task_data: List[TaskBase]):
        """Solve the task data (batch) using Pygmtools solver."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.GM:
            return gm_pygm_batch(
                batch_task_data=batch_task_data, 
                pygm_qap_solver=self.pygm_qap_solver
            )
        elif task_type == TASK_TYPE.GED:
            return ged_pygm_batch(
                batch_task_data=batch_task_data, 
                pygm_qap_solver=self.pygm_qap_solver
            )
        elif task_type == TASK_TYPE.KQAP:
            return kqap_pygm_batch(
                batch_task_data=batch_task_data, 
                pygm_qap_solver=self.pygm_qap_solver
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_type}."
            )