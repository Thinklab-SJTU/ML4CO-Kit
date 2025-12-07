r"""
GENN_AStar Solver.
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

import torch.nn as nn
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.astar.gm_genn_astar import gm_genn_astar
from ml4co_kit.solver.lib.astar.ged_genn_astar import ged_genn_astar
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class GennAStarSolver(SolverBase):
    def __init__(
        self,
        channel: int = None,
        filters_1: int = 64,
        filters_2: int = 32,
        filters_3: int = 16,
        tensor_neurons: int = 16,
        beam_width: int = 0,
        trust_fact: float = 1,  
        no_pred_size: int = 0,
        network: nn.Module = None,
        pretrain: str = "AIDS700nef",
        device: str = "cpu",
        optimizer: OptimizerBase = None
    ):
        super(GennAStarSolver, self).__init__(
            solver_type=SOLVER_TYPE.GENN_ASTAR, optimizer=optimizer
        )
        
        # Set Attributes
        self.channel = channel
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.tensor_neurons = tensor_neurons
        self.beam_width = beam_width  
        self.trust_fact = trust_fact
        self.no_pred_size = no_pred_size
        self.network = network
        self.pretrain = pretrain
        self.device = device
    
    def _batch_solve(self, task_data: list[TaskBase]):
        """Solve the task data using GNN_AStar solver."""
        if task_data[0].task_type == TASK_TYPE.GM:
            return gm_genn_astar(
                task_data=task_data,
                channel=self.channel,
                filters_1=self.filters_1,
                filters_2=self.filters_2,
                filters_3=self.filters_3,
                tensor_neurons=self.tensor_neurons,
                beam_width=self.beam_width,
                trust_fact=self.trust_fact,  
                no_pred_size=self.no_pred_size,
                network=self.network,
                pretrain=self.pretrain,
                device = self.device
            )
        elif task_data[0].task_type == TASK_TYPE.GED:
            return ged_genn_astar(
                task_data=task_data,
                channel=self.channel,
                filters_1=self.filters_1,
                filters_2=self.filters_2,
                filters_3=self.filters_3,
                tensor_neurons=self.tensor_neurons,
                beam_width=self.beam_width,
                trust_fact=self.trust_fact,  
                no_pred_size=self.no_pred_size,
                network=self.network,
                pretrain=self.pretrain,
                device = self.device
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )