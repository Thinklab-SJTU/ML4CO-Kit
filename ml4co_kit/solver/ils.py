r"""
ILS (Iterated Local Search) Solver.
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


from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.ils.pctsp_ils import pctsp_ils
from ml4co_kit.solver.lib.ils.spctsp_ils import spctsp_ils


class ILSSolver(SolverBase):
    def __init__(
        self, 
        ils_scale: int = 1e6,
        ils_runs: int = 1,
        spctsp_append_strategy: str = "half",
        optimizer: OptimizerBase = None
    ):
        super(ILSSolver, self).__init__(
            solver_type=SOLVER_TYPE.ILS, optimizer=optimizer
        )
        self.ils_scale = ils_scale
        self.ils_runs = ils_runs
        self.spctsp_append_strategy = spctsp_append_strategy
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using ILS Solver."""
        if task_data.task_type == TASK_TYPE.PCTSP:
            return pctsp_ils(
                task_data=task_data, 
                ils_scale=self.ils_scale,
                ils_runs=self.ils_runs
            )
        elif task_data.task_type == TASK_TYPE.SPCTSP:
            return spctsp_ils(
                task_data=task_data,
                ils_scale=self.ils_scale,
                ils_runs=self.ils_runs,
                spctsp_append_strategy=self.spctsp_append_strategy
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )