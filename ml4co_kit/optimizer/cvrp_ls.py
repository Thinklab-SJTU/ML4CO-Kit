r"""
CVRP Optimizer.
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
from ml4co_kit.optimizer.lib.cvrp_ls.cvrp_ls import cvrp_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE, IMPL_TYPE


class CVRPLSOptimizer(OptimizerBase):
    def __init__(
        self,
        impl_type: IMPL_TYPE = IMPL_TYPE.AUTO,
        coords_scale: int = 1000,
        demands_scale: int = 1000,
        seed: int = 1234
    ):
        # Super Initialization
        super(CVRPLSOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.CVRP_LS,
            impl_type=impl_type
        )
        
        # Set Attributes
        self.coords_scale = coords_scale
        self.demands_scale = demands_scale
        self.seed = seed
            
    def _auto_optimize(self, task_data: TaskBase, return_sol: bool = False):
        return self._ctypes_optimize(task_data, return_sol)

    def _ctypes_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using CVRP local search."""
        # Optimize
        task_type = task_data.task_type
        if task_type == TASK_TYPE.CVRP:
            cvrp_ls(
                task_data=task_data,
                coords_scale=self.coords_scale,
                demands_scale=self.demands_scale,
                seed=self.seed
            )
        else:
            raise self._get_not_implemented_error(task_type, False)

        # Return the solution if needed
        if return_sol:
            return task_data.sol
    
    def _auto_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the batch task data using auto implementation."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.CVRP:
            return self._ctypes_batch_optimize(batch_task_data)
        else:
            raise self._get_not_implemented_error(task_type, True)

    def _ctypes_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the batch task data using CVRP local search."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.CVRP:
            return self._pool_optimize(
                batch_task_data=batch_task_data,
                single_func=self._ctypes_optimize
            )
        else:
            raise self._get_not_implemented_error(task_type, True)