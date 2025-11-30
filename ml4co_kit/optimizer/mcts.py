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


from typing import List
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.lib.mcts.tsp_mcts import tsp_mcts_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE, IMPL_TYPE


class MCTSOptimizer(OptimizerBase):
    def __init__(
        self, 
        impl_type: IMPL_TYPE = IMPL_TYPE.AUTO,
        mcts_time_limit: float = 1.0,
        mcts_max_depth: int = 10, 
        mcts_type_2opt: int = 1,
        mcts_continue_flag: int = 2,
        mcts_max_iterations_2opt: int = 5000
    ):
        # Super Initialization
        super(MCTSOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.MCTS,
            impl_type=impl_type
        )
        
        # Set Attributes
        self.mcts_time_limit = mcts_time_limit
        self.mcts_max_depth = mcts_max_depth
        self.mcts_type_2opt = mcts_type_2opt
        self.mcts_continue_flag = mcts_continue_flag
        self.mcts_max_iterations_2opt = mcts_max_iterations_2opt

    def _auto_optimize(self, task_data: TaskBase, return_sol: bool = False):
        return self._ctypes_optimize(task_data, return_sol)

    def _ctypes_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using MCTS local search."""
        # Optimize
        task_type = task_data.task_type
        if task_type == TASK_TYPE.TSP:
            tsp_mcts_ls(
                task_data=task_data,
                mcts_time_limit=self.mcts_time_limit,
                mcts_max_depth=self.mcts_max_depth,
                mcts_type_2opt=self.mcts_type_2opt,
                mcts_continue_flag=self.mcts_continue_flag,
                mcts_max_iterations_2opt=self.mcts_max_iterations_2opt
            )
        else:
            raise self._get_not_implemented_error(task_type, False)
        
        # Return the solution if needed
        if return_sol:
            return task_data.sol
    
    def _auto_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the batch task data using auto implementation."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.TSP:
            return self._ctypes_batch_optimize(batch_task_data)
        else:
            raise self._get_not_implemented_error(task_type, True)

    def _ctypes_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the batch task data using MCTS local search."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.TSP:
            return self._pool_optimize(
                batch_task_data=batch_task_data,
                single_func=self._ctypes_optimize
            )
        else:
            raise self._get_not_implemented_error(task_type, True)