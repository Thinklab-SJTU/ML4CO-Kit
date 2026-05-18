r"""
PyVRP solver for routing problems.
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
from ml4co_kit.solver.routing.lib.pyvrp.cvrp_pyvrp import cvrp_pyvrp


class PyVRPSolver(SolverBase):
    """
    PyVRP solver.

    This solver currently supports CVRP only. The PyVRP package is imported
    lazily in the backend so importing ML4CO-Kit does not require PyVRP.
    """
    def __init__(
        self,
        pyvrp_time_limit: float = 1.0,
        pyvrp_seed: int = 1234,
        pyvrp_distance_scale: int = 10000,
        pyvrp_demand_scale: int = 10000,
        pyvrp_display: bool = False,
        optimizer: OptimizerBase = None,
    ):
        super(PyVRPSolver, self).__init__(
            solver_type=SOLVER_TYPE.PYVRP, optimizer=optimizer
        )

        # Set Attributes
        self.pyvrp_time_limit = pyvrp_time_limit
        self.pyvrp_seed = pyvrp_seed
        self.pyvrp_distance_scale = pyvrp_distance_scale
        self.pyvrp_demand_scale = pyvrp_demand_scale
        self.pyvrp_display = pyvrp_display

    def _solve(self, task_data: TaskBase):
        """Solve the task data using PyVRP solver."""
        if task_data.task_type == TASK_TYPE.CVRP:
            return cvrp_pyvrp(
                task_data=task_data,
                time_limit=self.pyvrp_time_limit,
                seed=self.pyvrp_seed,
                distance_scale=self.pyvrp_distance_scale,
                demand_scale=self.pyvrp_demand_scale,
                display=self.pyvrp_display,
            )
        else:
            raise ValueError(
                "PyVRPSolver only supports CVRP in the current implementation. "
                f"Got task type {task_data.task_type}."
            )
