r"""
Base class for all optimizers.
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


from enum import Enum
from multiprocessing import Pool
from typing import List, Callable, Dict
from ml4co_kit.utils.impl_utils import IMPL_TYPE
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class OPTIMIZER_TYPE(str, Enum):
    """Define the optimizer types as an enumeration."""
    
    # Routing Problems
    TWO_OPT = "two_opt"
    MCTS = "mcts"
    CVRP_LS = "cvrp_ls"
    
    # Graph Problems
    RLSA = "rlsa"


class OptimizerBase:
    """Base class for all optimizers."""
    
    def __init__(
        self, 
        optimizer_type: OPTIMIZER_TYPE,
        impl_type: IMPL_TYPE
    ):
        self.optimizer_type = optimizer_type
        self.impl_type = impl_type
        
        # Build method mapping for single optimization
        self._optimize_methods: Dict[IMPL_TYPE, Callable[[TaskBase, bool], None]] = {
            IMPL_TYPE.AUTO: self._auto_optimize,
            IMPL_TYPE.CTYPES: self._ctypes_optimize,
            IMPL_TYPE.CYTHON: self._cython_optimize,
            IMPL_TYPE.NUMPY: self._numpy_optimize,
            IMPL_TYPE.PYBIND11: self._pybind11_optimize,
            IMPL_TYPE.TORCH: self._torch_optimize,
        }
        
        # Build method mapping for batch optimization
        self._batch_optimize_methods: Dict[IMPL_TYPE, Callable[[List[TaskBase]], None]] = {
            IMPL_TYPE.AUTO: self._auto_batch_optimize,
            IMPL_TYPE.CTYPES: self._ctypes_batch_optimize,
            IMPL_TYPE.CYTHON: self._cython_batch_optimize,
            IMPL_TYPE.NUMPY: self._numpy_batch_optimize,
            IMPL_TYPE.PYBIND11: self._pybind11_batch_optimize,
            IMPL_TYPE.TORCH: self._torch_batch_optimize,
        }

    def optimize(self, task_data: TaskBase):
        """Optimize the given task data.
        
        Args:
            task_data: The task data to optimize.
            
        Raises:
            ValueError: If solution is None or implementation type is not supported.
        """
        # Check if solution is not None
        if task_data.sol is None:
            raise ValueError("`sol` cannot be None!")
        
        # Get the appropriate optimization method
        optimize_method = self._optimize_methods.get(self.impl_type, None)
        if optimize_method is None:
            raise ValueError(
                f"Implementation type {self.impl_type} is not supported."
            )
        
        # Optimize the task data
        optimize_method(task_data)

    def batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the given batch task data.
        
        Args:
            batch_task_data: List of task data to optimize.
            
        Raises:
            ValueError: If any solution is None or implementation type is not supported.
        """
        # Check if solution is not None
        if any(task_data.sol is None for task_data in batch_task_data):
            raise ValueError("`sol` cannot be None!")

        # Get the appropriate batch optimization method
        batch_optimize_method = self._batch_optimize_methods.get(self.impl_type, None)
        if batch_optimize_method is None:
            raise ValueError(
                f"Implementation type {self.impl_type} is not supported."
            )
        
        # Optimize the batch task data
        batch_optimize_method(batch_task_data)

    def _get_not_implemented_error(
        self, task_type: TASK_TYPE, batch: bool
    ) -> NotImplementedError:
        """Helper method to create a consistent NotImplementedError message."""
        if batch:
            return NotImplementedError(
                f"Optimizer {self.optimizer_type} with implementation type {self.impl_type} "
                f"is not supported for batch optimization of {task_type}."
            )
        else:
            return NotImplementedError(
                f"Optimizer {self.optimizer_type} with implementation type {self.impl_type} "
                f"is not supported for single optimization of {task_type}."
            )

    def _auto_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the given task data using auto implementation."""
        raise self._get_not_implemented_error(batch=False)
    
    def _ctypes_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the given task data using CTypes."""
        raise self._get_not_implemented_error(batch=False)

    def _cython_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the given task data using Cython."""
        raise self._get_not_implemented_error(batch=False)
    
    def _numpy_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the given task data using NumPy."""
        raise self._get_not_implemented_error(batch=False)
    
    def _pybind11_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the given task data using PyBind11."""
        raise self._get_not_implemented_error(batch=False)
        
    def _torch_optimize(self, task_data: TaskBase, return_sol: bool = False):
        """Optimize the given task data using Torch."""
        raise self._get_not_implemented_error(batch=False)

    def _auto_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the given batch task data using auto implementation."""
        raise self._get_not_implemented_error(batch=True)

    def _ctypes_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the given batch task data using CTypes."""
        raise self._get_not_implemented_error(batch=True)

    def _cython_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the given batch task data using Cython."""
        raise self._get_not_implemented_error(batch=True)
        
    def _numpy_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the given batch task data using NumPy."""
        raise self._get_not_implemented_error(batch=True)
        
    def _pybind11_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the given batch task data using PyBind11."""
        raise self._get_not_implemented_error(batch=True)
        
    def _torch_batch_optimize(self, batch_task_data: List[TaskBase]):
        """Optimize the given batch task data using Torch."""
        raise self._get_not_implemented_error(batch=True)

    def _pool_optimize(
        self, 
        batch_task_data: List[TaskBase], 
        single_func: Callable[[TaskBase, bool], None]
    ):
        """Optimize the given batch task data using Pool."""
        # Optimize parallelly
        with Pool(len(batch_task_data)) as p1:
            optimized_sols = p1.starmap(
                single_func, 
                [(task_data, True) for task_data in batch_task_data]
            )    

        # Update the original task data with the optimized solutions
        for task_data, sol in zip(batch_task_data, optimized_sols):
            task_data.sol = sol