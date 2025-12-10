r"""
NGM Solver.
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
from typing import Tuple
from torch import nn
from ml4co_kit.task.base import TASK_TYPE, TaskBase 
from ml4co_kit.solver.base import SOLVER_TYPE, SolverBase
from ml4co_kit.optimizer.base import OptimizerBase 
from ml4co_kit.solver.lib.ngm.ngm_function import ngm

class NGMSolver(SolverBase):
    def __init__(
        self,
        gnn_channels: Tuple[int, ...] = (16, 16, 16),
        sk_emb: int = 1,
        sk_max_iter: int = 50,
        sk_tau: float = 0.05,
        network: nn.Module = None,
        pretrain: str = "voc",
        device: str = "cpu",
        optimizer: OptimizerBase = None
    ):
        super().__init__(SOLVER_TYPE.NGM, optimizer=optimizer)
        
        self.gnn_channels = gnn_channels
        self.sk_emb = sk_emb
        self.sk_max_iter = sk_max_iter
        self.sk_tau = sk_tau
        self.network = network
        self.pretrain = pretrain
        self.device = device
        
    def _batch_solve(self, batch_task_data: list[TaskBase], x0: np.ndarray = None):
        """Solve the task data using NGM solver."""
        # Load model
        if batch_task_data[0].task_type == TASK_TYPE.GM:
            return ngm(
                batch_task_data=batch_task_data,
                x0=x0,
                gnn_channels=self.gnn_channels,
                sk_emb=self.sk_emb,
                sk_max_iter=self.sk_max_iter,
                sk_tau=self.sk_tau,
                network=self.network,
                pretrain=self.pretrain,
                device=self.device
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {batch_task_data[0].task_type}."
            )