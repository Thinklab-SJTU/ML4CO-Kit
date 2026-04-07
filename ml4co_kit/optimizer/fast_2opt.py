r"""
Fast 2-Opt optimizer.
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
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE, IMPL_TYPE
from ml4co_kit.optimizer.lib.fast_2opt.tsp_fast_2opt import pybind11_tsp_fast_2opt_ls


class FastTwoOptOptimizer(OptimizerBase):
    """Fast 2-opt."""

    def __init__(
        self,
        impl_type: IMPL_TYPE = IMPL_TYPE.AUTO,
        num_steps: int = 1e6,
        knn: int = 50,
        seed: int = 1234,
        num_workers: int = 1
    ):
        # Super Initialization
        super(FastTwoOptOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.FAST_2OPT,
            impl_type=impl_type
        )

        # Set Attributes
        self.num_steps = num_steps
        self.knn = knn
        self.seed = seed
        self.num_workers = num_workers

    #######################################
    #     Single Optimization Methods     #
    #######################################

    def _auto_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using auto implementation."""
        if task_data.task_type == TASK_TYPE.TSP:
            return self._pybind11_optimize(task_data, return_sol)
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} ({self.impl_type})"
                f"is not supported for {task_data.task_type}."
            )
    
    def _pybind11_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using DC 2-opt (pybind11)."""
        # Optimize
        task_type = task_data.task_type
        if task_type == TASK_TYPE.TSP:
            pybind11_tsp_fast_2opt_ls(
                task_data=task_data,
                num_steps=self.num_steps,
                knn=self.knn,
                seed=self.seed,
            )
        else:
            raise self._get_not_implemented_error(task_type, False)

        # Return the solution if needed
        if return_sol:
            return task_data.sol