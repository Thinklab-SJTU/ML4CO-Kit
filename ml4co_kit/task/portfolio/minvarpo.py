r"""
Minimum Variance Portfolio Optimization (MinVarPO)
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


class MinVarPOTask(PortfolioTaskBase):
    def __init__(
        self, 
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(MinVarPOTask, self).__init__(
            task_type=TASK_TYPE.MINVARPO, 
            minimize=True, 
            precision=precision,
            threshold=threshold
        )

        # Initialize Attributes
        self.required_returns = None   # Required returns

    def _check_required_returns_not_none(self):
        """Check if required returns is not None."""
        if self.required_returns is None:
            raise ValueError("required returns cannot be None!")

    def from_data(
        self,
        returns: np.ndarray = None,
        cov: np.ndarray = None,
        required_returns: float = None,
        sol: np.ndarray = None,
        ref: bool = False,
        name: str = None
    ):
        # Call Super Method ``from_data``
        super().from_data(
            returns=returns, cov=cov, sol=sol, ref=ref, name=name
        )

        # Set Attributes
        if required_returns is not None:
            self.required_returns = required_returns

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        # Check Constraints of Base Class (01)
        base_check = super().check_constraints(sol)
        if not base_check:
            return False

        # Constraint-02: Required returns
        sum_returns = np.sum(self.returns * sol)
        if sum_returns < self.required_returns - self.threshold:
            return False

        return True

    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the given solution."""
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")

        # risk-only: minimize portfolio variance w^T Σ w
        variance = np.dot(sol, self.cov @ sol)
        return variance

    def render(self):
        """Render the MinVarPO problem instance with or without solution."""
        raise NotImplementedError("Render is not implemented for MinVarPO.")