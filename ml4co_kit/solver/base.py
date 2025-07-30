import numpy as np
from typing import Union
from ml4co_kit.utils.type_utils import TASK_TYPE, SOLVER_TYPE


class SolverBase(object):
    def __init__(
        self, 
        task_type: TASK_TYPE = None, 
        solver_type: SOLVER_TYPE = None,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        self.task_type = task_type
        self.solver_type = solver_type
        self.precision = precision
        self.solve_msg = f"Solving {self.task_type} Using {self.solver_type}"
    
    def from_txt(self, *args, **kwargs):
        raise NotImplementedError(
            "The ``from_txt`` function is required to implemented in subclasses."
        )
        
    def to_txt(self, *args, **kwargs):
        raise NotImplementedError(
            "The ``to_txt`` function is required to implemented in subclasses."
        )
        
    def solve(self, *args, **kwargs):
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "SolverBase"