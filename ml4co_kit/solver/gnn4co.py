r"""
Greedy Solver.
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


import torch
from typing import List
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.extension.gnn4co.model.model import GNN4COModel


class GNN4COSolver(SolverBase):
    def __init__(
        self, 
        model: GNN4COModel, 
        device: str = "cpu",
        optimizer: OptimizerBase = None
    ):
        super(GNN4COSolver, self).__init__(
            solver_type=SOLVER_TYPE.GNN4CO, optimizer=optimizer
        )
        self.device = device
        self.model = model
        self.model.model.to(self.device)
        self.model.env.device = self.device
        self.model.env.data_processor.device = self.device

    def _solve(self, task_data: TaskBase):
        """Solve the task data using Greedy Solver."""
        return self._batch_solve([task_data])

    def _batch_solve(self, batch_task_data: List[TaskBase]):
        """Solve the task data using Greedy Solver."""
        # Process data
        processor = self.model.env.data_processor
        batch_processed_data = processor.batch_data_process(batch_task_data)

        # Inference
        task_type = batch_task_data[0].task_type
        with torch.no_grad():
            if self.model.env.sparse:
                if task_type == TASK_TYPE.TSP:
                    heatmap = self.model.inference_edge_sparse(batch_processed_data)
                    self.model.decoder.sparse_decode(
                        heatmap=heatmap, 
                        task_type=task_type,
                        batch_task_data=batch_task_data,
                        batch_processed_data=batch_processed_data
                    )
                elif task_type in [TASK_TYPE.MIS, TASK_TYPE.MCUT, TASK_TYPE.MCL, TASK_TYPE.MVC]:
                    heatmap = self.model.inference_node_sparse(batch_processed_data)
                    self.model.decoder.sparse_decode(
                        heatmap=heatmap, 
                        task_type=task_type,
                        batch_task_data=batch_task_data,
                        batch_processed_data=batch_processed_data
                    )
                else:
                    raise NotImplementedError()
            else:
                if task_type in [TASK_TYPE.TSP, TASK_TYPE.ATSP]:
                    heatmap = self.model.inference_edge_dense(batch_processed_data)
                    self.model.decoder.dense_decode(
                        heatmap=heatmap, 
                        task_type=task_type,batch_task_data=batch_task_data,
                        batch_processed_data=batch_processed_data
                    )
                else:
                    raise NotImplementedError()