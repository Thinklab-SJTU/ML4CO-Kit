r"""
Maximum Return Portfolio Optimization (MaxRetPO)
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
from ml4co_kit.task.portfolio.maxretpo import MaxRetPOTask
from ml4co_kit.generator.portfolio.base import (
    PortfolioGeneratorBase, PO_TYPE, PortfolioDistributionArgs
)


class MaxRetPOGenerator(PortfolioGeneratorBase):
    def __init__(
        self, 
        distribution_type: PO_TYPE = PO_TYPE.GBM,
        precision: Union[np.float32, np.float64] = np.float32,
        assets_num: int = 50,
        time_steps: int = 252,
        distribution_args: PortfolioDistributionArgs = PortfolioDistributionArgs(),
        max_var: float = None,
        max_var_ratio: float = 0.7
    ):
        # Super Initialization
        super(MaxRetPOGenerator, self).__init__(
            task_type=TASK_TYPE.MAXRETPO,
            distribution_type=distribution_type,
            precision=precision,
            assets_num=assets_num,
            time_steps=time_steps,
            distribution_args=distribution_args
        )

        # Initialize Attributes
        self.max_var = max_var
        self.max_var_ratio = max_var_ratio

    def _create_task(self, returns: np.ndarray, cov: np.ndarray) -> MaxRetPOTask:
        if self.max_var is None:
            self.max_var = np.max(np.diag(cov)) * self.max_var_ratio
        data = MaxRetPOTask(precision=self.precision)
        data.from_data(
            returns=returns, cov=cov, max_var=self.max_var 
        )
        return data