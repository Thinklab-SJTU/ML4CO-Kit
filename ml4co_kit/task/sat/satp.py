r"""
Satisfiability Prediction (SAP-P).
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
from typing import List, Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.sat.base import SATTaskBase


class SATPTask(SATTaskBase):
    """Satisfiability Prediction (SAP-P) task."""
    
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(SATPTask, self).__init__(
            task_type=TASK_TYPE.SATP, 
            minimize=False, 
            precision=precision
        )

        # Initialize Attributes
        self.satisfiable: bool = None

    def _check_sol_dim(self):
        """Ensure solution is a boolean."""
        if self.sol not in [True, False]:
            raise ValueError("Solution should be a boolean.")

    def _check_ref_sol_dim(self):
        """Ensure reference solution is a boolean."""
        if self.ref_sol not in [True, False]:
            raise ValueError("Reference solution should be a boolean.")
    
    def from_data(
        self, 
        clauses: List[List[int]], 
        vars_num: int = None, 
        satisfiable: bool = None,
        sol: bool = None, 
        ref: bool = False, 
        name: str = None
    ):
        # Super Initialization
        super(SATPTask, self).from_data(
            clauses=clauses, vars_num=vars_num, 
            sol=sol, ref=ref, name=name
        )

        # Set Satisfiability if Provided
        if satisfiable is not None:
            self.satisfiable = satisfiable

    def check_constraints(self, sol: bool) -> bool:
        return True # for boolean solution, no constraints are needed

    def evaluate(self, sol: bool) -> np.floating:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Check Satisfiability
        if self.satisfiable is None:
            raise ValueError("Satisfiability is not set!")
        
        # Evaluate
        result = 1.0 if sol == self.satisfiable else 0.0
        return np.array(result).astype(self.precision)