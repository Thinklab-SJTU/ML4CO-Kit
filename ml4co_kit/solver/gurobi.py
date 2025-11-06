r"""
Gurobi Solver.
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
from ml4co_kit.solver.lib.gurobi.op_gurobi import op_gurobi
from ml4co_kit.solver.lib.gurobi.tsp_gurobi import tsp_gurobi
from ml4co_kit.solver.lib.gurobi.mcl_gurobi import mcl_gurobi
from ml4co_kit.solver.lib.gurobi.mis_gurobi import mis_gurobi
from ml4co_kit.solver.lib.gurobi.mvc_gurobi import mvc_gurobi
from ml4co_kit.solver.lib.gurobi.atsp_gurobi import atsp_gurobi
from ml4co_kit.solver.lib.gurobi.cvrp_gurobi import cvrp_gurobi
from ml4co_kit.solver.lib.gurobi.mcut_gurobi import mcut_gurobi
from ml4co_kit.solver.lib.gurobi.mopo_gurobi import mopo_gurobi
from ml4co_kit.solver.lib.gurobi.maxretpo_gurobi import maxretpo_gurobi
from ml4co_kit.solver.lib.gurobi.minvarpo_gurobi import minvarpo_gurobi


class GurobiSolver(SolverBase):
    def __init__(
        self, 
        gurobi_time_limit: float = 10.0, 
        gurobi_tsp_use_mtz_or_lazy: str = "lazy",
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(GurobiSolver, self).__init__(
            solver_type=SOLVER_TYPE.GUROBI, optimizer=optimizer
        )
        
        # Set Attributes
        self.gurobi_time_limit = gurobi_time_limit
        self.gurobi_tsp_use_mtz_or_lazy = gurobi_tsp_use_mtz_or_lazy

    def _solve(self, task_data: TaskBase):
        """Solve the task data using Gurobi Solver."""
        if task_data.task_type == TASK_TYPE.ATSP:
            return atsp_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.CVRP:
            return cvrp_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.OP:
            return op_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.TSP:
            return tsp_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit,
                gurobi_tsp_use_mtz_or_lazy=self.gurobi_tsp_use_mtz_or_lazy
            )
        elif task_data.task_type == TASK_TYPE.MCL:
            return mcl_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MCUT:
            return mcut_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MIS:
            return mis_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MAXRETPO:
            return maxretpo_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MINVARPO:
            return minvarpo_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MOPO:
            return mopo_gurobi(
                task_data=task_data,
                gurobi_time_limit=self.gurobi_time_limit
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )