r"""
Concorde Solver.
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


import os
import pathlib
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.concorde.tsp_concorde import tsp_concorde


class ConcordeSolver(SolverBase):
    def __init__(
        self,
        concorde_scale: int = 1e6,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(ConcordeSolver, self).__init__(
            solver_type=SOLVER_TYPE.CONCORDE, optimizer=optimizer
        )

        # Initialize Attributes
        self.concorde_scale = concorde_scale

        # Check if need re-compile
        try:
            from ml4co_kit.solver.lib.concorde.pyconcorde import TSPConSolver
        except:
            self.install()
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using LKH solver."""
        if task_data.task_type == TASK_TYPE.TSP:
            return tsp_concorde(
                task_data=task_data,
                concorde_scale=self.concorde_scale,
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )

    def install(self):
        """Install Concorde solver."""
        concorde_path = pathlib.Path(__file__).parent / "lib/concorde/pyconcorde"
        ori_dir = os.getcwd()
        os.chdir(concorde_path)
        os.system("python ./setup.py build_ext --inplace")
        os.chdir(ori_dir)