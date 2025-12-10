r"""
Unsat-core Variable Prediction (USAT-C).
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.sat.base import SATTaskBase


class USATCTask(SATTaskBase):
    """Unsat-core Variable Prediction (USAT-C) task."""
    
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(USATCTask, self).__init__(
            task_type=TASK_TYPE.USATC, 
            minimize=False, 
            precision=precision
        )

        # Initialize Attributes
        self.satisfiable: bool = False
        self.unsat_core_vars: np.ndarray = None

    def _check_sol_dim(self):
        """Ensure solution is a 1D array."""
        if self.sol.ndim != 1 or self.sol.dtype != bool:
            raise ValueError("Solution should be a 1D boolean array.")

    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 1D boolean array."""
        if self.ref_sol.ndim != 1 or self.ref_sol.dtype != bool:
            raise ValueError("Reference solution should be a 1D boolean array.")

    def check_constraints(self, sol: np.ndarray) -> bool:
        # Check Dimensions
        if len(sol) != self.vars_num:
            return False
        
        # Check if all values are 0 or 1
        return np.all((sol == 0) | (sol == 1))

    def evaluate(self, sol: np.ndarray) -> np.floating:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Check Unsat-core Variables
        if self.unsat_core_vars is None:
            raise ValueError("Unsat-core variables are not set!")
        
        # Check if all unsat-core variables are in the solution
        ratio = np.mean(self.unsat_core_vars == sol)
        return np.array(ratio).astype(self.precision)