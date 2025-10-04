r"""
MCTS Optimizer.
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


from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.lib.mcts.tsp_mcts import tsp_mcts_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE


class MCTSOptimizer(OptimizerBase):
    def __init__(
        self, 
        mcts_time_limit: float = 1.0,
        mcts_max_depth: int = 10, 
        mcts_type_2opt: int = 1,
        mcts_continue_flag: int = 2,
        mcts_max_iterations_2opt: int = 5000
    ):
        # Super Initialization
        super(MCTSOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.MCTS
        )
        
        # Set Attributes
        self.mcts_time_limit = mcts_time_limit
        self.mcts_max_depth = mcts_max_depth
        self.mcts_type_2opt = mcts_type_2opt
        self.mcts_continue_flag = mcts_continue_flag
        self.mcts_max_iterations_2opt = mcts_max_iterations_2opt
            
    def _optimize(self, task_data: TaskBase):
        """Optimize the task data using MCTS local search."""
        if task_data.task_type == TASK_TYPE.TSP:
            return tsp_mcts_ls(
                task_data=task_data,
                mcts_time_limit=self.mcts_time_limit,
                mcts_max_depth=self.mcts_max_depth,
                mcts_type_2opt=self.mcts_type_2opt,
                mcts_continue_flag=self.mcts_continue_flag,
                mcts_max_iterations_2opt=self.mcts_max_iterations_2opt
            )
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} "
                f"is not supported for {task_data.task_type}."
            )