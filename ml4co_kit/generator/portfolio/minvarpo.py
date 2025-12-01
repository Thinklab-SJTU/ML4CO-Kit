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
from ml4co_kit.task.portfolio.minvarpo import MinVarPOTask
from ml4co_kit.generator.portfolio.base import (
    PortfolioGeneratorBase, PO_TYPE, PortfolioDistributionArgs
)


class MinVarPOGenerator(PortfolioGeneratorBase):
    def __init__(
        self, 
        distribution_type: PO_TYPE = PO_TYPE.GBM,
        precision: Union[np.float32, np.float64] = np.float32,
        assets_num: int = 50,
        time_steps: int = 252,
        distribution_args: PortfolioDistributionArgs = PortfolioDistributionArgs(),
        required_returns: float = None,
        required_returns_ratio: float = 0.7
    ):
        # Super Initialization
        super(MinVarPOGenerator, self).__init__(
            task_type=TASK_TYPE.MINVARPO,
            distribution_type=distribution_type,
            precision=precision,
            assets_num=assets_num,
            time_steps=time_steps,
            distribution_args=distribution_args
        )

        # Initialize Attributes
        self.required_returns = required_returns
        self.required_returns_ratio = required_returns_ratio

    def _create_task(self, returns: np.ndarray, cov: np.ndarray) -> MinVarPOTask:
        if self.required_returns is None:
            mean_returns = np.mean(returns)
            self.required_returns = mean_returns * self.required_returns_ratio
        data = MinVarPOTask(precision=self.precision)
        data.from_data(
            returns=returns, cov=cov, required_returns=self.required_returns
        )
        return data