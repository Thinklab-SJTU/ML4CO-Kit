r"""
ISCO Optimizer.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import Callable
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.lib.isco.mcl_isco import mcl_isco_ls
from ml4co_kit.optimizer.lib.isco.mis_isco import mis_isco_ls
from ml4co_kit.optimizer.lib.isco.mvc_isco import mvc_isco_ls
from ml4co_kit.optimizer.lib.isco.mcut_isco import mcut_isco_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE


class ISCOOptimizer(OptimizerBase):
    def __init__(
        self,
        isco_tau: float = 0.5, 
        isco_mu_init: float = 5.0,
        isco_g_func: Callable[[np.ndarray], np.ndarray] = lambda r: np.sqrt(r),
        isco_adapt_mu: bool = True,
        isco_target_accept_rate: float = 0.574,
        isco_beta: float = 1.002,
        isco_iterations: int = 10000,
        isco_seed: int = 1234,
    ):
        # Super Initialization
        super(ISCOOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.RLSA
        )
        
        # Set Attributes
        self.isco_tau = isco_tau
        self.isco_mu_init = isco_mu_init
        self.isco_g_func = isco_g_func
        self.isco_adapt_mu = isco_adapt_mu
        self.isco_target_accept_rate = isco_target_accept_rate
        self.isco_beta = isco_beta
        self.isco_iterations = isco_iterations
        self.isco_seed = isco_seed
            
    def _optimize(self, task_data: TaskBase):
        """Optimize the task data using RLSA local search."""
        if task_data.task_type == TASK_TYPE.MCL:
            return mcl_isco_ls(
                task_data=task_data,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_beta=self.isco_beta,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        elif task_data.task_type == TASK_TYPE.MCUT:
            return mcut_isco_ls(
                task_data=task_data,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        elif task_data.task_type == TASK_TYPE.MIS:
            return mis_isco_ls(
                task_data=task_data,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_beta=self.isco_beta,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_isco_ls(
                task_data=task_data,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_beta=self.isco_beta,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} "
                f"is not supported for {task_data.task_type}."
            )