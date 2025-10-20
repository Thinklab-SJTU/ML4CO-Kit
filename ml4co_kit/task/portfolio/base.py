r"""
Base class for all portfolio optimization problems.
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


class PortfolioTaskBase(TaskBase):
    def __init__(
        self, 
        task_type: TASK_TYPE,
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(PortfolioTaskBase, self).__init__(
            task_type=task_type, 
            minimize=minimize, 
            precision=precision
        )

        # Initialize Attributes
        self.returns: np.ndarray = None        # Returns of assets
        self.cov: np.ndarray = None            # Covariance matrix of assets
        self.num_assets: int = None            # Number of assets
        self.threshold: float = threshold      # Threshold for floating point precision

    def _check_returns_dim(self):
        """Check if returns is a 1D array."""
        if self.returns.ndim != 1:
            raise ValueError(
                "returns should be a 2D array with shape (num_assets,)."
            )
        
    def _check_returns_not_none(self):
        """Check if returns is not None."""
        if self.returns is None:
            raise ValueError("returns cannot be None!")

    def _check_cov_dim(self):
        """Check if cov is a 2D array."""
        if self.cov.ndim != 2 and self.cov.shape[1] != self.num_assets:
            raise ValueError(
                "cov should be a 2D array with shape (num_assets, num_assets)."
            )

    def _check_cov_not_none(self):
        """Check if cov is not None."""
        if self.cov is None:
            raise ValueError("cov cannot be None!")

    def _check_cov_symmetric(self):
        """Check if cov is symmetric."""
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError("cov should be a symmetric matrix.")

    def _check_sol_dim(self):
        """Ensure solution is a 1D array."""
        if self.sol.ndim != 1:
            raise ValueError("Solution should be a 1D array.")
        
    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 1D array."""
        if self.ref_sol.ndim != 1:
            raise ValueError("Reference solution should be a 1D array.")

    def from_data(
        self,
        returns: np.ndarray = None,
        cov: np.ndarray = None,
        sol: np.ndarray = None,
        ref: bool = False,
        name: str = None
    ):
        # Set Attributes and Check Dimensions
        if returns is not None:
            self.returns = returns.astype(self.precision)
            self._check_returns_dim()

        if cov is not None:
            self.cov = cov.astype(self.precision)
            self._check_cov_dim()
            self._check_cov_symmetric()
        
        if sol is not None:
            if ref:
                self.ref_sol = sol.astype(self.precision)
                self._check_ref_sol_dim()
            else:
                self.sol = sol.astype(self.precision)
                self._check_sol_dim()

        # Set Number of Assets if Provided
        if self.returns is not None:
            self.num_assets = self.returns.shape[0]

        # Set Name if Provided
        if name is not None:
            self.name = name

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        # Constraint: weights must sum to 1
        sol_num = np.sum(sol)
        if not np.isclose(
            a=sol_num, 
            b=np.array(1.0).astype(self.precision), 
            atol=float(self.threshold)
        ):
            return False
        return True