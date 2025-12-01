r"""
Multi-Objective Portfolio Optimization (MOPO)
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
from ml4co_kit.task.portfolio.base import PortfolioTaskBase


class MOPOTask(PortfolioTaskBase):
    def __init__(
        self, 
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(MOPOTask, self).__init__(
            task_type=TASK_TYPE.MOPO, 
            minimize=True, 
            precision=precision,
            threshold=threshold
        )

        # Initialize Attributes
        self.var_factor = None         # Variance factor
        self.ret_factor = None         # Return factor

    def _check_var_factor_not_none(self):
        """Check if required returns is not None."""
        if self.var_factor is None:
            raise ValueError("required returns cannot be None!")
            
    def from_data(
        self,
        returns: np.ndarray = None,
        cov: np.ndarray = None,
        var_factor: float = None,
        sol: np.ndarray = None,
        ref: bool = False,
        name: str = None
    ):
        # Call Super Method ``from_data``
        super().from_data(
            returns=returns, cov=cov, sol=sol, ref=ref, name=name
        )

        # Set Attributes
        if var_factor is not None:
            self.var_factor = var_factor
            self.ret_factor = 1 - var_factor

    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the given solution."""
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")

        # minimize portfolio variance w^T Σ w and
        # maximize portfolio returns r^T w
        variance = np.dot(sol, self.cov @ sol)
        returns = np.sum(self.returns * sol)
        obj = variance * self.var_factor - returns * self.ret_factor
        return obj

    def render(self):
        """Render the MOPO problem instance with or without solution."""
        raise NotImplementedError("Render is not implemented for MOPO.")