r"""
ORToolsi Solver.
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


from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver.routing_enums_pb2 import LocalSearchMetaheuristic
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.ortools.op_ortools import op_ortools
from ml4co_kit.solver.lib.ortools.tsp_ortools import tsp_ortools
from ml4co_kit.solver.lib.ortools.mcl_ortools import mcl_ortools
from ml4co_kit.solver.lib.ortools.mis_ortools import mis_ortools
from ml4co_kit.solver.lib.ortools.mvc_ortools import mvc_ortools
from ml4co_kit.solver.lib.ortools.atsp_ortools import atsp_ortools
from ml4co_kit.solver.lib.ortools.pctsp_ortools import pctsp_ortools


class ORSolver(SolverBase):
    def __init__(
        self, 
        ortools_scale: int = 1e6,
        ortools_time_limit: int = 10,
        routing_ls_strategy: str = "guided",        
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(ORSolver, self).__init__(
            solver_type=SOLVER_TYPE.ORTOOLS, optimizer=optimizer
        )
        
        # Set Attributes
        self.ortools_scale = ortools_scale
        self.ortools_time_limit = ortools_time_limit
        self.routing_ls_strategy = routing_ls_strategy

    def _set_search_parameters(self):
        """Set the search parameters for the OR Solver."""
        meta_heu_dict = {
            "auto": LocalSearchMetaheuristic.AUTOMATIC,
            "greedy": LocalSearchMetaheuristic.GREEDY_DESCENT,
            "guided": LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
            "simulated": LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            "tabu": LocalSearchMetaheuristic.TABU_SEARCH,
            "generic_tabu": LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
        }
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.search_parameters.local_search_metaheuristic = \
            (meta_heu_dict[self.routing_ls_strategy])
        self.search_parameters.time_limit.seconds = self.ortools_time_limit
    
    def _solve(self, task_data: TaskBase):
        """Solve the task data using OR Solver."""
        if task_data.task_type == TASK_TYPE.ATSP:
            self._set_search_parameters()
            return atsp_ortools(
                task_data=task_data,
                ortools_scale=self.ortools_scale,
                search_parameters=self.search_parameters
            )
        elif task_data.task_type == TASK_TYPE.PCTSP:
            self._set_search_parameters()
            return pctsp_ortools(
                task_data=task_data,
                ortools_scale=self.ortools_scale,
                search_parameters=self.search_parameters
            )
        elif task_data.task_type == TASK_TYPE.OP:
            self._set_search_parameters()
            return op_ortools(
                task_data=task_data,
                ortools_scale=self.ortools_scale,
                search_parameters=self.search_parameters
            )
        elif task_data.task_type == TASK_TYPE.TSP:
            self._set_search_parameters()
            return tsp_ortools(
                task_data=task_data,
                ortools_scale=self.ortools_scale,
                search_parameters=self.search_parameters
            )
        elif task_data.task_type == TASK_TYPE.MCL:
            return mcl_ortools(
                task_data=task_data,
                ortools_scale=self.ortools_scale,
                ortools_time_limit=self.ortools_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MIS:
            return mis_ortools(
                task_data=task_data,
                ortools_scale=self.ortools_scale,
                ortools_time_limit=self.ortools_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MVC:
            return mvc_ortools(
                task_data=task_data,
                ortools_scale=self.ortools_scale,
                ortools_time_limit=self.ortools_time_limit
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )