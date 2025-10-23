r"""
Base class for logic problem generators in the ML4CO kit.
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
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.generator.base import GeneratorBase


class LOGIC_TYPE(str, Enum):
    """Define the logic problem types as an enumeration."""
    RANDOM = "random"           # Random SAT instances
    UNIFORM_RANDOM = "uniform_random"  # Uniform random k-SAT
    PLANTED = "planted"         # Planted solutions
    PHASE_TRANSITION = "phase_transition"  # Near phase transition
    INDUSTRIAL = "industrial"   # Industrial-like instances


class LogicGeneratorBase(GeneratorBase):
    """Base class for logic problem generators."""
    
    def __init__(
        self,
        task_type: TASK_TYPE,
        distribution_type: LOGIC_TYPE,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=task_type,
            distribution_type=distribution_type,
            precision=precision
        )
        
        # Logic-specific attributes
        self.num_vars: int = 10         # Default number of variables
        self.num_clauses: int = 42      # Default number of clauses
        self.clause_length: int = 3     # Default clause length (k in k-SAT)
        
        # Random seed for reproducibility
        self.seed: int = None
        
    def set_parameters(
        self,
        num_vars: int,
        num_clauses: int = None,
        clause_length: int = 3,
        seed: int = None
    ):
        """Set generation parameters."""
        self.num_vars = num_vars
        self.clause_length = clause_length
        self.seed = seed
        
        # Auto-calculate num_clauses if not provided
        if num_clauses is None:
            # Use SAT-UNSAT phase transition ratio for k-SAT
            if clause_length == 2:
                ratio = 1.0     # 2-SAT phase transition
            elif clause_length == 3:
                ratio = 4.26    # 3-SAT phase transition
            elif clause_length == 4:
                ratio = 9.93    # 4-SAT phase transition
            else:
                ratio = 2**(clause_length) * np.log(2)  # General approximation
            
            self.num_clauses = int(ratio * num_vars)
        else:
            self.num_clauses = num_clauses
    
    def _set_random_seed(self):
        """Set random seed if provided."""
        if self.seed is not None:
            np.random.seed(self.seed)