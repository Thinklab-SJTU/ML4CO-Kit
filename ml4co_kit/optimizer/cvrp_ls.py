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


from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.lib.cvrp_ls.cvrp_ls import cvrp_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE


class CVRPLSOptimizer(OptimizerBase):
    def __init__(
        self,
        coords_scale: int = 1000,
        demands_scale: int = 1000,
        seed: int = 1234
    ):
        # Super Initialization
        super(CVRPLSOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.CVRP_LS
        )
        
        # Set Attributes
        self.coords_scale = coords_scale
        self.demands_scale = demands_scale
        self.seed = seed
            
    def _optimize(self, task_data: TaskBase):
        """Optimize the task data using CVRP local search."""
        if task_data.task_type == TASK_TYPE.CVRP:
            return cvrp_ls(
                task_data=task_data,
                coords_scale=self.coords_scale,
                demands_scale=self.demands_scale,
                seed=self.seed
            )
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} "
                f"is not supported for {task_data.task_type}."
            )