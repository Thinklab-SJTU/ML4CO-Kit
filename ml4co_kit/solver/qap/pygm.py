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
from ml4co_kit.extension.pygmtools import PyGMToolsQAPSolver
from .lib.pygm.gm_pygm import gm_pygm, gm_pygm_batch
from .lib.pygm.ged_pygm import ged_pygm, ged_pygm_batch
from .lib.pygm.kqap_pygm import kqap_pygm, kqap_pygm_batch


class PyGMSolver(SolverBase):
    """
    Pygmtools: https://github.com/Thinklab-SJTU/Pygmtools
    Current Version: 0.6.0
    Last Update: 2026-05-26
    @article{
        wang2024pygm,
        author  = {Runzhong Wang and Ziao Guo and Wenzheng Pan and Jiale Ma and 
            Yikai Zhang and Nan Yang and Qi Liu and Longxuan Wei and Hanxue Zhang and 
            Chang Liu and Zetian Jiang and Xiaokang Yang and Junchi Yan},
        title   = {Pygmtools: A Python Graph Matching Toolkit},
        journal = {Journal of Machine Learning Research},
        year    = {2024},
        volume  = {25},
        number  = {33},
        pages   = {1-7},
        url     = {https://jmlr.org/papers/v25/23-0572.html},
    }
    """
    def __init__(
        self,
        temperature: float = 0.3,
        pygm_qap_solver: PyGMToolsQAPSolver = PyGMToolsQAPSolver(),
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(PyGMSolver, self).__init__(
            solver_type=SOLVER_TYPE.PYGM, optimizer=optimizer
        )

        # Set Attributes
        self.pygm_qap_solver = pygm_qap_solver
        self.temperature = temperature

    def _solve(self, task_data: TaskBase):
        """Solve the task data using Pygmtools solver."""
        if task_data.task_type == TASK_TYPE.GM:
            return gm_pygm(
                task_data=task_data, pygm_qap_solver=self.pygm_qap_solver
            )
        elif task_data.task_type == TASK_TYPE.GED:
            return ged_pygm(
                task_data=task_data, pygm_qap_solver=self.pygm_qap_solver, temperature=self.temperature
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