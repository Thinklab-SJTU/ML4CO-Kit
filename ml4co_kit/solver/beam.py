r"""
Beam Solver.
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
from ml4co_kit.utils.type_utils import to_numpy
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.extension.gnn4co.model.model import GNN4COModel
from ml4co_kit.solver.lib.beam.mcl_beam import mcl_beam
from ml4co_kit.solver.lib.beam.mis_beam import mis_beam


class BeamSolver(SolverBase):
    def __init__(
        self, 
        model: GNN4COModel, 
        device: str = "cpu",
        optimizer: OptimizerBase = None
    ):
        super(BeamSolver, self).__init__(
            solver_type=SOLVER_TYPE.GREEDY, optimizer=optimizer
        )
        self.device = device
        self.model = model
        self.model.model.to(self.device)
        self.model.env.change_device(self.device)

    def _solve(self, task_data: TaskBase):
        """Solve the task data using Beam Solver."""
        # Using ``data_process`` to process task data
        data = self.model.env.data_processor.data_process([task_data])
        
        # Inference to get heatmap
        if task_data.task_type in [
            TASK_TYPE.MIS, TASK_TYPE.MCUT, TASK_TYPE.MCL, TASK_TYPE.MVC
        ]:
            with torch.no_grad():
                heatmap = self.model.inference_node_sparse_process(*data)
            task_data.cache["heatmap"] = to_numpy(heatmap)
        elif task_data.task_type in [
            TASK_TYPE.ATSP, TASK_TYPE.CVRP, TASK_TYPE.TSP
        ]:
            if self.model.env.sparse:
                with torch.no_grad():
                    heatmap = self.model.inference_edge_sparse_process(*data)
            else:
                with torch.no_grad():
                    heatmap = self.model.inference_edge_dense_process(*data)    
            task_data.cache["heatmap"] = to_numpy(heatmap[0])
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )
                 
        # Solve task data
        if task_data.task_type == TASK_TYPE.MIS:
            return mis_beam(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MCL:
            return mcl_beam(task_data=task_data)
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )