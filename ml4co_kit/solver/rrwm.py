r"""
RRWM(Random Reweighted Walk for Graph Matching) Solver.
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
from ml4co_kit.solver.lib.rrwm.gm_rrwm import gm_rrwm
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE




class RRWMSolver(SolverBase):
    def __init__(
        self,
        x0: np.ndarray = None,
        max_iter: int = 50,
        sk_iter: int = 20,
        alpha: float = 0.2,
        beta: float = 30.0,
        optimizer: OptimizerBase = None
    ):
        super(RRWMSolver, self).__init__(
            solver_type=SOLVER_TYPE.RRWM, optimizer=optimizer
        )
        
        # Set Attributes
        self.x0 = x0
        self.max_iter = max_iter
        self.sk_iter = sk_iter
        self.alpha = alpha
        self.beta = beta
    
    def _solve(self, task_data: TaskBase):
        """Solve the task data using RRWM solver."""
        if task_data.task_type == TASK_TYPE.GM:
            return gm_rrwm(
                task_data=task_data,
                x0 = self.x0,
                max_iter=self.max_iter,
                sk_iter=self.sk_iter,
                alpha=self.alpha,
                beta=self.beta
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )