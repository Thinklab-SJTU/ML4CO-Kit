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
from ml4co_kit.task.portfolio.mopo import MOPOTask
from ml4co_kit.generator.portfolio.base import (
    PortfolioGeneratorBase, PO_TYPE, PortfolioDistributionArgs
)


class MOPOGenerator(PortfolioGeneratorBase):
    def __init__(
        self, 
        distribution_type: PO_TYPE = PO_TYPE.GBM,
        precision: Union[np.float32, np.float64] = np.float32,
        assets_num: int = 50,
        time_steps: int = 252,
        distribution_args: PortfolioDistributionArgs = PortfolioDistributionArgs(),
        var_factor: float = 0.7
    ):
        # Super Initialization
        super(MOPOGenerator, self).__init__(
            task_type=TASK_TYPE.MOPO,
            distribution_type=distribution_type,
            precision=precision,
            assets_num=assets_num,
            time_steps=time_steps,
            distribution_args=distribution_args
        )

        # Initialize Attributes
        self.var_factor = var_factor

    def _create_task(self, returns: np.ndarray, cov: np.ndarray) -> MOPOTask:
        data = MOPOTask(precision=self.precision)
        data.from_data(returns=returns, cov=cov, var_factor=self.var_factor)
        return data