r"""
Two-opt Optimizer.
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
from ml4co_kit.optimizer.lib.two_opt.tsp_2opt import tsp_2opt_ls
from ml4co_kit.optimizer.lib.two_opt.atsp_2opt import atsp_2opt_ls
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE


class TwoOptOptimizer(OptimizerBase):
    def __init__(
        self, max_iters: int = 5000, device: str = "cpu"
    ):
        # Super Initialization
        super(TwoOptOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.TWO_OPT
        )
        
        # Set Attributes
        self.max_iters = max_iters
        self.device = device
            
    def _optimize(self, task_data: TaskBase):
        """Optimize the task data using 2-opt local search."""
        if task_data.task_type == TASK_TYPE.TSP:
            return tsp_2opt_ls(
                task_data=task_data,
                max_iters=self.max_iters,
                device=self.device
            )
        elif task_data.task_type == TASK_TYPE.ATSP:
            return atsp_2opt_ls(
                task_data=task_data,
                max_iters=self.max_iters,
            )
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} "
                f"is not supported for {task_data.task_type}."
            )