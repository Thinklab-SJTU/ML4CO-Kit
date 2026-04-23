r"""
FEM (Free Energy Minimization) Solver.
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


import math
import torch
from functools import partial
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.fem.mcut_fem import mcut_fem


class FEMSolver(SolverBase):
    """
    FEM: https://github.com/Fanerst/FEM.
    @article{
        shen2025free,
        title={Free-energy machine for combinatorial optimization},
        author={Shen, Zi-Song and Pan, Feng and Wang, Yao and Men, 
        Yi-Ding and Xu, Wen-Biao and Yung, Man-Hong and Zhang, Pan},
        journal={Nature Computational Science},
        volume={5},
        number={4},
        pages={322--332},
        year={2025},
        publisher={Nature Publishing Group US New York}
    }
    """
    
    def __init__(
        self,
        num_trials: int = 100,
        num_steps: int = 1000,
        beta_range: tuple = (0.01, 0.5),
        anneal_type: str = "inverse",
        grad_opt_type: str = "rmsprop",
        lr: float = 0.1,
        device: str = "cpu",
        seed: int = 1,
        h_factor: float = 0.01,
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(FEMSolver, self).__init__(
            solver_type=SOLVER_TYPE.FEM,
            optimizer=optimizer
        )
        
        # Set Attributes
        self.num_trials = num_trials
        self.device = device
        self.seed = seed
        self.h_factor = h_factor

        # Get betas
        beta_min, beta_max = beta_range
        if anneal_type == "inverse":
            betas = 1.0 / torch.linspace(beta_max, beta_min, num_steps)
        elif anneal_type == "linear":
            betas = torch.linspace(beta_min, beta_max, num_steps)
        elif anneal_type == "exp":
            log_beta_min = math.log(beta_min)
            log_beta_max = math.log(beta_max)
            log_betas = torch.linspace(log_beta_min, log_beta_max, num_steps)
            betas = torch.exp(log_betas)
        else:
            raise ValueError(f"Unknown anneal type: {anneal_type}")
        self.betas = betas

        # Gradient optimizer
        if grad_opt_type == "adam":
            grad_opt_class = partial(torch.optim.Adam, lr=lr)
        elif grad_opt_type == "rmsprop":
            grad_opt_class = partial(
                torch.optim.RMSprop, lr=lr, alpha=0.98, eps=1e-08, 
                weight_decay=0.01, momentum=0.91, centered=False
            )
        else:
            raise ValueError(f"Unknown gradient optimizer type: {grad_opt_type}")
        self.grad_opt_class = grad_opt_class
    
    def _solve(self, task_data: TaskBase):
        """Solve the task data using FEM Solver."""
        if task_data.task_type == TASK_TYPE.MCUT:
            return mcut_fem(
                task_data=task_data,
                num_trials=self.num_trials,
                betas=self.betas,
                grad_opt_class=self.grad_opt_class,
                device=self.device,
                seed=self.seed,
                h_factor=self.h_factor
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )