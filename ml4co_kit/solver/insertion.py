r"""
Insertion (Manual Heuristic)
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
from ml4co_kit.solver.lib.insertion.tsp_insertion import tsp_insertion
from ml4co_kit.solver.lib.insertion.mcl_insertion import mcl_insertion
from ml4co_kit.solver.lib.insertion.mis_insertion import mis_insertion
from ml4co_kit.solver.lib.insertion.mvc_insertion import mvc_insertion
from ml4co_kit.solver.lib.insertion.mcut_insertion import mcut_insertion


class InsertionSolver(SolverBase):
    """
    Insertion-TSP: https://github.com/henry-yeh/GLOP
    @inproceedings{
        ye2024glop,
        title={GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time},
        author={Ye, Haoran and Wang, Jiarui and Liang, Helan and Cao, Zhiguang and Li, Yong and Li, Fanzhang},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2024},
    }
    """
    def __init__(self, optimizer: OptimizerBase = None):
        super(InsertionSolver, self).__init__(
            solver_type=SOLVER_TYPE.INSERTION, optimizer=optimizer
        )
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using LKH solver."""
        if task_data.task_type == TASK_TYPE.TSP:
            return tsp_insertion(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MCL:
            return mcl_insertion(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MIS:
            return mis_insertion(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_insertion(task_data=task_data)
        elif task_data.task_type == TASK_TYPE.MCUT:
            return mcut_insertion(task_data=task_data)
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )