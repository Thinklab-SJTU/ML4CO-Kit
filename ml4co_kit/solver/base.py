r"""
Base class for all solvers.
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
from typing import List
from ml4co_kit.task.base import TaskBase
from ml4co_kit.optimizer.base import OptimizerBase


class SOLVER_TYPE(str, Enum):
    """Define the solver types as an enumeration."""
    # Does not need any dependencies
    CONCORDE = "concorde"
    GA_EAX = "ga_eax"
    GP_DEGREE = "gp_degree"
    HGS = "hgs"
    ILS = "ils"
    INSERTION = "insertion"
    KAMIS = "kamis"
    LC_DEGREE = "lc_degree"
    LKH = "lkh"
    ORTOOLS = "ortools"
    PYVRP = "pyvrp"
    SCIP = "scip"

    # Need Gurobi License
    GUROBI = "gurobi"

    # Need Torch
    BEAM = "beam"
    GREEDY = "greedy"
    ISCO = "isco"
    MCTS = "mcts"
    NEUROLKH = "neurolkh"
    RLSA = "rlsa"


class SolverBase(object):
    """Base class for all solvers."""
    
    def __init__(
        self, 
        solver_type: SOLVER_TYPE,
        optimizer: OptimizerBase = None,
    ):
        self.solver_type = solver_type   
        self.solve_func_dict: dict = None
        self.optimizer = optimizer
    
    def solve(self, task_data: TaskBase) -> TaskBase:
        self._solve(task_data)
        if self.optimizer is not None:
            self.optimizer.optimize(task_data)
        return task_data
    
    def _solve(self, task_data: TaskBase):
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )
    
    def batch_solve(self, batch_task_data: List[TaskBase]) -> List[TaskBase]:
        self._batch_solve(batch_task_data)
        if self.optimizer is not None:
            for task_data in batch_task_data:
                self.optimizer.optimize(task_data)
        return batch_task_data
    
    def _batch_solve(self, batch_task_data: List[TaskBase]):
        raise NotImplementedError(
            "The ``batch_solve`` function is required to implemented in subclasses."
        )