r"""
MCMC (Markov Chain Monte Carlo) Optimizer
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


import random
import numpy as np
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.lib.mcmc.mis_mcmc import mis_mcmc_ls
from ml4co_kit.optimizer.lib.mcmc.tsp_mcmc import tsp_mcmc_ls
from ml4co_kit.optimizer.lib.mcmc.cvrp_mcmc import cvrp_mcmc_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE, IMPL_TYPE


class MCMCOptimizer(OptimizerBase):
    """
    MCMC Optimizer.
    """

    def __init__(
        self,
        impl_type: IMPL_TYPE = IMPL_TYPE.AUTO,
        tau_strategy: str = "linear",
        tau_start: float = 1.0,
        tau_end: float = 0.001,
        num_steps: int = 100000,
        seed: int = 1234,
        penalty_coeff: float = 1.02
    ):
        # Super Initialization
        super(MCMCOptimizer, self).__init__(
            OPTIMIZER_TYPE.MCMC, impl_type=impl_type
        )

        # Set Attributes
        self.taus = self._get_times(
            tau_strategy=tau_strategy,
            tau_start=tau_start,
            tau_end=tau_end,
            num_steps=int(num_steps)
        )
        self.seed = seed
        self.penalty_coeff = penalty_coeff

    def _get_times(
        self, 
        tau_strategy: str, 
        tau_start: float, 
        tau_end: float, 
        num_steps: int
    ) -> np.ndarray:
        """Get the times for the MCMC solver."""
        if tau_strategy == "linear":
            return np.linspace(tau_start, tau_end, num_steps)
        else:
            raise ValueError(f"Invalid tau strategy: {tau_strategy}")

    #######################################
    #     Single Optimization Methods     #
    #######################################

    def _auto_optimize(self, task_data: TaskBase, return_sol: bool = False):
        return self._pybind11_optimize(task_data, return_sol)

    def _pybind11_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Solve the task data using MCMC optimizer."""
        if task_data.task_type == TASK_TYPE.MIS:
            mis_mcmc_ls(
                task_data=task_data,
                taus=self.taus,
                penalty_coeff=self.penalty_coeff,
                seed=self.seed,
            )
        elif task_data.task_type == TASK_TYPE.TSP:
            tsp_mcmc_ls(
                task_data=task_data,
                taus=self.taus,
                seed=self.seed,
            )
        elif task_data.task_type == TASK_TYPE.CVRP:
            cvrp_mcmc_ls(
                task_data=task_data,
                taus=self.taus,
                penalty_coeff=self.penalty_coeff,
                seed=self.seed,
            )
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} is not supported for {task_data.task_type}."
            )

        # Return the solution if needed
        if return_sol:
            return task_data.sol