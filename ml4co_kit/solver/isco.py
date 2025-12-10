r"""
ISCO (Improved Sampling Algorithm for Combinatorial Optimization)
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
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.isco.mcl_isco import mcl_isco
from ml4co_kit.solver.lib.isco.mis_isco import mis_isco
from ml4co_kit.solver.lib.isco.mvc_isco import mvc_isco
from ml4co_kit.solver.lib.isco.mcut_isco import mcut_isco
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class ISCOSolver(SolverBase):
    """
    DISCS: https://github.com/google-research/discs
    @article{
        goshvadi2023discs,
        title={Discs: a benchmark for discrete sampling},
        author={Goshvadi, Katayoon and Sun, Haoran and Liu, Xingchao and Nova, \
            Azade and Zhang, Ruqi and Grathwohl, Will and Schuurmans, Dale and Dai, Hanjun},
        journal={Advances in Neural Information Processing Systems},
        volume={36},
        pages={79035--79066},
        year={2023}
    }
    @inproceedings{
        sun2023revisiting,
        title={Revisiting sampling for combinatorial optimization},
        author={Sun, Haoran and Goshvadi, Katayoon and Nova, Azade and Schuurmans, Dale and Dai, Hanjun},
        booktitle={International Conference on Machine Learning},
        pages={32859--32874},
        year={2023},
        organization={PMLR}
    }
    """
    def __init__(
        self, 
        isco_init_type: str = "uniform",
        isco_tau: float = 0.5, 
        isco_mu_init: float = 5.0,
        isco_g_func: Callable[[np.ndarray], np.ndarray] = lambda r: np.sqrt(r),
        isco_adapt_mu: bool = True,
        isco_target_accept_rate: float = 0.574,
        isco_alpha: float = 0.3,
        isco_beta: float = 1.002,
        isco_iterations: int = 10000,
        isco_seed: int = 1234,
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(ISCOSolver, self).__init__(SOLVER_TYPE.ISCO, optimizer=optimizer)
        
        # Set Attributes
        self.isco_init_type = isco_init_type
        self.isco_tau = isco_tau
        self.isco_mu_init = isco_mu_init
        self.isco_g_func = isco_g_func
        self.isco_adapt_mu = isco_adapt_mu
        self.isco_target_accept_rate = isco_target_accept_rate
        self.isco_alpha = isco_alpha
        self.isco_beta = isco_beta
        self.isco_iterations = isco_iterations
        self.isco_seed = isco_seed
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using ISCO solver."""
        if task_data.task_type == TASK_TYPE.MCL:
            return mcl_isco(
                task_data=task_data,
                isco_init_type=self.isco_init_type,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_alpha=self.isco_alpha,
                isco_beta=self.isco_beta,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        elif task_data.task_type == TASK_TYPE.MCUT:
            return mcut_isco(
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
            return mis_isco(
                task_data=task_data,
                isco_init_type=self.isco_init_type,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_alpha=self.isco_alpha,
                isco_beta=self.isco_beta,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_isco(
                task_data=task_data,
                isco_init_type=self.isco_init_type,
                isco_tau=self.isco_tau,
                isco_mu_init=self.isco_mu_init,
                isco_g_func=self.isco_g_func,
                isco_adapt_mu=self.isco_adapt_mu,
                isco_target_accept_rate=self.isco_target_accept_rate,
                isco_alpha=self.isco_alpha,
                isco_beta=self.isco_beta,
                isco_iterations=self.isco_iterations,
                isco_seed=self.isco_seed
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )