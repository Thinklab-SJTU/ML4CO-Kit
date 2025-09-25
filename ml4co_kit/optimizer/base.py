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
from ml4co_kit.task.base import TaskBase


class OPTIMIZER_TYPE(str, Enum):
    """Define the optimizer types as an enumeration."""
    
    # Routing Problems
    TWO_OPT = "two_opt"
    MCTS = "mcts"
    CVRP_LS = "cvrp_ls"
    
    # Graph Problems
    RLSA = "rlsa"


class OptimizerBase(object):
    """Base class for all optimizers."""
    
    def __init__(self, optimizer_type: OPTIMIZER_TYPE):
        self.optimizer_type = optimizer_type
    
    def optimize(self, task_data: TaskBase):
        """Optimize the given task data."""
        # Check if solution is not None
        if task_data.sol is None:
            raise ValueError("``sol`` cannot be None!")
        
        # Optimize the task data
        self._optimize(task_data)
    
    def _optimize(self, task_data: TaskBase):
        """Optimize the given task data."""
        raise NotImplementedError("Subclasses should implement this method.")