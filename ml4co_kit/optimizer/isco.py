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
from typing import Callable, List
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.lib.isco.mcl_isco import mcl_isco_ls
from ml4co_kit.optimizer.lib.isco.mis_isco import mis_isco_ls
from ml4co_kit.optimizer.lib.isco.mvc_isco import mvc_isco_ls
from ml4co_kit.optimizer.lib.isco.mcut_isco import mcut_isco_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE, IMPL_TYPE


class ISCOOptimizer(OptimizerBase):
    def __init__(
        self,
        impl_type: IMPL_TYPE = IMPL_TYPE.AUTO,
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
            optimizer_type=OPTIMIZER_TYPE.RLSA,
            impl_type=impl_type
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
            
    def _auto_optimize(self, task_data: TaskBase, return_sol: bool = False):
        return self._numpy_optimize(task_data, return_sol)

    def _numpy_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using ISCO local search."""
        # Optimize
        task_type = task_data.task_type
        if task_type == TASK_TYPE.MCL:
            mcl_isco_ls(
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
        elif task_type == TASK_TYPE.MCUT:
            mcut_isco_ls(
                task_data=task_data,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        elif task_type == TASK_TYPE.MIS:
            mis_isco_ls(
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
        elif task_type == TASK_TYPE.MVC:
            mvc_isco_ls(
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
            raise self._get_not_implemented_error(task_type, False)

        # Return the solution if needed
        if return_sol:
            return task_data.sol
        
    def _auto_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the batch task data using auto implementation."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.MCL:
            return self._numpy_batch_optimize(batch_task_data)
        else:
            raise self._get_not_implemented_error(task_type, True)

    def _numpy_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the batch task data using ISCO local search."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.MCL:
            return self._pool_optimize(
                batch_task_data=batch_task_data,
                single_func=self._numpy_optimize
            )
        else:
            raise self._get_not_implemented_error(task_type, True)