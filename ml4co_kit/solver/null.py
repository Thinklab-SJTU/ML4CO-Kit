r"""
Null Solver: Do nothing.
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
from ml4co_kit.task.base import TaskBase
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class NullSolver(SolverBase):
    """
    Null Solver: Do nothing.
    """
    def __init__(
        self,
        cp_from_ref: bool = True,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(NullSolver, self).__init__(
            solver_type=SOLVER_TYPE.NULL, optimizer=optimizer
        )

        # Initialize Attributes
        self.cp_from_ref = cp_from_ref

    def _solve(self, task_data: TaskBase):
        if self.cp_from_ref:
            variable = task_data.ref_sol
            if isinstance(variable, np.ndarray):    
                task_data.sol = variable.copy()
            else:
                task_data.sol = variable