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


from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.mcts.tsp_mcts import tsp_mcts
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class MCTSSolver(SolverBase):
    def __init__(
        self, 
        model, 
        mcts_time_limit: float = 1.0,
        mcts_max_depth: int = 10, 
        mcts_type_2opt: int = 1, 
        mcts_max_iterations_2opt: int = 5000,
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(MCTSSolver).__init__(SOLVER_TYPE.MCTS, optimizer=optimizer)
        
        # Set Attributes
        self.model = model
        self.mcts_time_limit = mcts_time_limit
        self.mcts_max_depth = mcts_max_depth
        self.mcts_type_2opt = mcts_type_2opt
        self.mcts_max_iterations_2opt = mcts_max_iterations_2opt
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using LKH solver."""
        # Get heatmap 
        heatmap = self.model.encode(task_data)
        task_data.cache["heatmap"] = heatmap
        
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