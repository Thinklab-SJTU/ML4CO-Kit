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


from typing import List
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.optimizer.base import OptimizerBase, OPTIMIZER_TYPE, IMPL_TYPE
from ml4co_kit.optimizer.lib.two_opt.atsp.ctypes_atsp_2opt import ctypes_atsp_2opt_ls
from ml4co_kit.optimizer.lib.two_opt.tsp.torch_tsp_2opt import torch_tsp_2opt_ls
from ml4co_kit.optimizer.lib.two_opt.tsp.torch_tsp_2opt import torch_tsp_2opt_batch_ls
from ml4co_kit.optimizer.lib.two_opt.tsp.pybind11_tsp_2opt import pybind11_tsp_2opt_ls
from ml4co_kit.optimizer.lib.two_opt.tsp.pybind11_tsp_2opt import pybind11_tsp_2opt_batch_ls


class TwoOptOptimizer(OptimizerBase):
    def __init__(
        self, 
        impl_type: IMPL_TYPE = IMPL_TYPE.AUTO,
        max_iters: int = 5000, 
        torch_device: str = "cpu",
        type_2opt: int = 2,
    ):
        # Super Initialization
        super(TwoOptOptimizer, self).__init__(
            optimizer_type=OPTIMIZER_TYPE.TWO_OPT,
            impl_type=impl_type
        )
        
        # Set Attributes
        self.max_iters = max_iters
        self.torch_device = torch_device
        self.type_2opt = type_2opt

    #######################################
    #     Single Optimization Methods     #
    #######################################

    def _auto_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using auto implementation."""
        if task_data.task_type == TASK_TYPE.ATSP:
            return self._ctypes_optimize(task_data, return_sol)
        elif task_data.task_type == TASK_TYPE.TSP:
            return self._pybind11_optimize(task_data, return_sol)
        else:
            raise ValueError(
                f"Optimizer {self.optimizer_type} ({self.impl_type})"
                f"is not supported for {task_data.task_type}."
            )

    def _ctypes_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using 2-opt local search (ctypes)."""
        # Optimize
        task_type = task_data.task_type
        if task_type == TASK_TYPE.ATSP:
            ctypes_atsp_2opt_ls(
                task_data=task_data,
                max_iters=self.max_iters
            )
        else:
            raise self._get_not_implemented_error(task_type, False)

        # Return the solution if needed
        if return_sol:
            return task_data.sol
    
    def _pybind11_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using 2-opt local search (pybind11)."""
        # Optimize
        task_type = task_data.task_type
        if task_type == TASK_TYPE.TSP:
            pybind11_tsp_2opt_ls(
                task_data=task_data,
                max_iters=self.max_iters,
                type_2opt=self.type_2opt
            )
        else:
            raise self._get_not_implemented_error(task_type, False)

        # Return the solution if needed
        if return_sol:
            return task_data.sol
    
    def _torch_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the task data using 2-opt local search (torch)."""
        # Optimize
        task_type = task_data.task_type
        if task_type == TASK_TYPE.TSP:
            torch_tsp_2opt_ls(
                task_data=task_data,
                max_iters=self.max_iters,
                device=self.torch_device
            )
        else:
            raise self._get_not_implemented_error(task_type, False)

        # Return the solution if needed
        if return_sol:
            return task_data.sol

    #######################################
    #     Batch Optimization Methods     #
    #######################################

    def _auto_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the (batch) task data using auto implementation."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.ATSP:
            return self._ctypes_batch_optimize(batch_task_data)
        elif task_type == TASK_TYPE.TSP:
            return self._torch_batch_optimize(batch_task_data)
        else:
            raise NotImplementedError(
                f"Optimizer {self.optimizer_type} is not supported for {task_type}."
            )

    def _ctypes_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the (batch) task data using 2-opt local search (ctypes)."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.ATSP:
            return self._pool_optimize(
                batch_task_data=batch_task_data,
                single_func=self._ctypes_optimize
            )
        else:
            raise self._get_not_implemented_error(task_type, True)

    def _pybind11_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the (batch) task data using 2-opt local search (pybind11)."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.TSP:  
            return pybind11_tsp_2opt_batch_ls(
                    batch_task_data=batch_task_data,
                    max_iters=self.max_iters,
                    type_2opt=self.type_2opt
                )
        else:
            raise self._get_not_implemented_error(task_type, True)

    def _torch_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the (batch) task data using 2-opt local search (torch)."""
        task_type = batch_task_data[0].task_type
        if task_type == TASK_TYPE.TSP:
            return torch_tsp_2opt_batch_ls(
                batch_task_data=batch_task_data,
                max_iters=self.max_iters,
                device=self.torch_device
            )
        else:
            raise self._get_not_implemented_error(task_type, True)