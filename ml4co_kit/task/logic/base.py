r"""
Base class for logic problems in the ML4CO kit.
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
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class LogicTaskBase(TaskBase):
    """Base class for logic problems."""
    
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool = False,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=task_type, 
            minimize=minimize, 
            precision=precision
        )
        
        # Logic-specific attributes
        self.num_vars: int = 0                    # Number of variables
        self.clauses: list = []                   # List of clauses
        self.assignment: np.ndarray = None        # Current variable assignment
        self.ref_assignment: np.ndarray = None    # Reference assignment
    
    def get_num_vars(self) -> int:
        """Get the number of variables."""
        return self.num_vars
    
    def get_num_clauses(self) -> int:
        """Get the number of clauses."""
        return len(self.clauses)
    
    def set_assignment(self, assignment: np.ndarray):
        """Set the variable assignment."""
        if len(assignment) != self.num_vars:
            raise ValueError(f"Assignment length {len(assignment)} != num_vars {self.num_vars}")
        self.assignment = assignment.astype(np.bool_)
        # Also set sol for compatibility with base class
        self.sol = assignment.astype(np.int32)
    
    def set_ref_assignment(self, ref_assignment: np.ndarray):
        """Set the reference variable assignment."""
        if len(ref_assignment) != self.num_vars:
            raise ValueError(f"Reference assignment length {len(ref_assignment)} != num_vars {self.num_vars}")
        self.ref_assignment = ref_assignment.astype(np.bool_)
        # Also set ref_sol for compatibility with base class
        self.ref_sol = ref_assignment.astype(np.int32)