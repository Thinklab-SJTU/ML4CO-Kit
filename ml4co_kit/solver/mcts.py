r"""
MCTS Solver.
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
from ml4co_kit.solver.lib.mcts.tsp_mcts import tsp_mcts
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.extension.gnn4co.model.model import GNN4COModel


class MCTSSolver(SolverBase):
    def __init__(
        self, 
        model: GNN4COModel,
        mcts_time_limit: float = 1.0,
        mcts_max_depth: int = 10, 
        mcts_type_2opt: int = 1, 
        mcts_max_iterations_2opt: int = 5000,
        device: str = "cpu",
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(MCTSSolver, self).__init__(SOLVER_TYPE.MCTS, optimizer=optimizer)
        
        # Set Attributes for Model
        self.device = device
        self.model = model
        self.model.model.to(self.device)
        self.model.env.change_device(self.device)
        
        # Set Attributes for MCTS
        self.mcts_time_limit = mcts_time_limit
        self.mcts_max_depth = mcts_max_depth
        self.mcts_type_2opt = mcts_type_2opt
        self.mcts_max_iterations_2opt = mcts_max_iterations_2opt
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using LKH solver."""
        # Using ``data_process`` to process task data
        data = self.model.env.data_processor.data_process([task_data])        

        # Get heatmap 
        if self.model.env.sparse:
            with torch.no_grad():
                heatmap = self.model.inference_edge_sparse_process(*data)
        else:
            with torch.no_grad():
                heatmap = self.model.inference_edge_dense_process(*data)    
        task_data.cache["heatmap"] = to_numpy(heatmap[0])
        
        # Solve task data
        if task_data.task_type == TASK_TYPE.TSP:
            return tsp_mcts(
                task_data=task_data,
                mcts_time_limit=self.mcts_time_limit,
                mcts_max_depth=self.mcts_max_depth,
                mcts_type_2opt=self.mcts_type_2opt,
                mcts_max_iterations_2opt=self.mcts_max_iterations_2opt,
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )