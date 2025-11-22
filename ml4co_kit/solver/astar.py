r"""
AStar Solver.
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

import numpy as np
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.astar.gm_classical_astar import gm_astar
from ml4co_kit.solver.lib.astar.ged_classical_astar import ged_astar
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class AStarSolver(SolverBase):
    def __init__(
        self,
        beam_width: int = 0,
        optimizer: OptimizerBase = None
    ):
        super(AStarSolver, self).__init__(
            solver_type=SOLVER_TYPE.ASTAR, optimizer=optimizer
        )
        
        # Set Attributes
        self.beam_width = beam_width   
    
    def _solve(self, task_data: TaskBase, beam_width: int = 0):
        """Solve the task data using AStar solver."""
        if task_data.task_type == TASK_TYPE.GM:
            return gm_astar(
                task_data=task_data,
                beam_width=self.beam_width,
            )
        elif task_data.task_type == TASK_TYPE.GED:
            return ged_astar(
                task_data=task_data,
                beam_width=self.beam_width,
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )