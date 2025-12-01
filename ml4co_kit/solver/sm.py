r"""
SM(Spectral Matching) Solver.
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

import numpy as np
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.sm.gm_sm import gm_sm
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class SMSolver(SolverBase):
    def __init__(
        self,
        x0: np.ndarray = None,
        max_iter: int = 50,
        optimizer: OptimizerBase = None
    ):
        super(SMSolver, self).__init__(
            solver_type=SOLVER_TYPE.SM, optimizer=optimizer
        )
        
        # Set Attributes
        self.x0 = x0
        self.max_iter = max_iter   
    
    def _solve(self, task_data: TaskBase):
        """Solve the task data using SM solver."""
        if task_data.task_type == TASK_TYPE.GM:
            return gm_sm(
                task_data=task_data,
                x0 = self.x0,
                max_iter = self.max_iter
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )