r"""
RLSA Optimizer.
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


from typing import Union
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.lib.rlsa.mcl_rlsa import mcl_rlsa_ls
from ml4co_kit.optimizer.lib.rlsa.mis_rlsa import mis_rlsa_ls
from ml4co_kit.optimizer.lib.rlsa.mvc_rlsa import mvc_rlsa_ls
from ml4co_kit.optimizer.lib.rlsa.mcut_rlsa import mcut_rlsa_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE


class RLSAOptimizer(OptimizerBase):
    def __init__(
        self,
        rlsa_init_type: str = "uniform",
        rlsa_kth_dim: Union[str, int] = "both",
        rlsa_tau: float = 0.01, 
        rlsa_d: int = 5, 
        rlsa_k: int = 1000, 
        rlsa_t: int = 1000, 
        rlsa_alpha: float = 0.3,
        rlsa_beta: float = 1.02,
        rlsa_device: str = "cpu", 
        rlsa_seed: int = 1234
    ):
        # Super Initialization
        super(RLSAOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.RLSA
        )
        
        # Set Attributes
        self.rlsa_init_type = rlsa_init_type
        self.rlsa_kth_dim = rlsa_kth_dim
        self.rlsa_tau = rlsa_tau
        self.rlsa_d = rlsa_d
        self.rlsa_k = rlsa_k
        self.rlsa_t = rlsa_t
        self.rlsa_alpha = rlsa_alpha
        self.rlsa_beta = rlsa_beta
        self.rlsa_device = rlsa_device
        self.rlsa_seed = rlsa_seed
            
    def _optimize(self, task_data: TaskBase):
        """Optimize the task data using RLSA local search."""
        if task_data.task_type == TASK_TYPE.MCL:
            return mcl_rlsa_ls(
                task_data=task_data,
                rlsa_init_type=self.rlsa_init_type,
                rlsa_kth_dim=self.rlsa_kth_dim,
                rlsa_tau=self.rlsa_tau,
                rlsa_d=self.rlsa_d,
                rlsa_k=self.rlsa_k,
                rlsa_t=self.rlsa_t,
                rlsa_alpha=self.rlsa_alpha,
                rlsa_beta=self.rlsa_beta,
                rlsa_device=self.rlsa_device,
                rlsa_seed=self.rlsa_seed
            )
        elif task_data.task_type == TASK_TYPE.MCUT:
            return mcut_rlsa_ls(
                task_data=task_data,
                rlsa_kth_dim=self.rlsa_kth_dim,
                rlsa_tau=self.rlsa_tau,
                rlsa_d=self.rlsa_d,
                rlsa_k=self.rlsa_k,
                rlsa_t=self.rlsa_t,
                rlsa_device=self.rlsa_device,
                rlsa_seed=self.rlsa_seed
            )
        elif task_data.task_type == TASK_TYPE.MIS:
            return mis_rlsa_ls(
                task_data=task_data,
                rlsa_init_type=self.rlsa_init_type,
                rlsa_kth_dim=self.rlsa_kth_dim,
                rlsa_tau=self.rlsa_tau,
                rlsa_d=self.rlsa_d,
                rlsa_k=self.rlsa_k,
                rlsa_t=self.rlsa_t,
                rlsa_alpha=self.rlsa_alpha,
                rlsa_beta=self.rlsa_beta,
                rlsa_device=self.rlsa_device,
                rlsa_seed=self.rlsa_seed
            )
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_rlsa_ls(
                task_data=task_data,
                rlsa_init_type=self.rlsa_init_type,
                rlsa_kth_dim=self.rlsa_kth_dim,
                rlsa_tau=self.rlsa_tau,
                rlsa_d=self.rlsa_d,
                rlsa_k=self.rlsa_k,
                rlsa_t=self.rlsa_t,
                rlsa_alpha=self.rlsa_alpha,
                rlsa_beta=self.rlsa_beta,
                rlsa_device=self.rlsa_device,
                rlsa_seed=self.rlsa_seed
            )
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} "
                f"is not supported for {task_data.task_type}."
            )