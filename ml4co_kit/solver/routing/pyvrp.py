r"""
PyVRP solver for CVRP variants.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from .lib.pyvrp.cvrp_pyvrp import cvrp_pyvrp
from .lib.pyvrp.cvrpb_pyvrp import cvrpb_pyvrp
from .lib.pyvrp.cvrpbl_pyvrp import cvrpbl_pyvrp
from .lib.pyvrp.cvrpbltw_pyvrp import cvrpbltw_pyvrp
from .lib.pyvrp.cvrpbtw_pyvrp import cvrpbtw_pyvrp
from .lib.pyvrp.cvrpl_pyvrp import cvrpl_pyvrp
from .lib.pyvrp.cvrpltw_pyvrp import cvrpltw_pyvrp
from .lib.pyvrp.cvrptw_pyvrp import cvrptw_pyvrp


class PyVRPSolver(SolverBase):
    """
    PyVRP: https://github.com/PyVRP/PyVRP
    Current Version: v0.13.4
    Last Update: 2026-05-26
    @article{
        Wouda_Lan_Kool_PyVRP_2024,
        doi = {10.1287/ijoc.2023.0055},
        url = {https://doi.org/10.1287/ijoc.2023.0055},
        year = {2024},
        volume = {36},
        number = {4},
        pages = {943--955},
        publisher = {INFORMS},
        author = {Niels A. Wouda and Leon Lan and Wouter Kool},
        title = {{PyVRP}: a high-performance {VRP} solver package},
        journal = {INFORMS Journal on Computing},
    }
    """
    def __init__(
        self,
        time_limit: float = 1.0,
        scale: int = int(1e5),
        seed: int = 1234,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(PyVRPSolver, self).__init__(
            solver_type=SOLVER_TYPE.PYVRP, optimizer=optimizer
        )

        # Set Attributes
        self.time_limit = time_limit
        self.scale = scale
        self.seed = seed

    def _solve(self, task_data: TaskBase):
        """Solve the task data using PyVRP solver."""
        if task_data.task_type == TASK_TYPE.CVRP:
            return cvrp_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        elif task_data.task_type == TASK_TYPE.CVRPB:
            return cvrpb_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        elif task_data.task_type == TASK_TYPE.CVRPBL:
            return cvrpbl_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        elif task_data.task_type == TASK_TYPE.CVRPBLTW:
            return cvrpbltw_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        elif task_data.task_type == TASK_TYPE.CVRPBTW:
            return cvrpbtw_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        elif task_data.task_type == TASK_TYPE.CVRPL:
            return cvrpl_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        elif task_data.task_type == TASK_TYPE.CVRPLTW:
            return cvrpltw_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        elif task_data.task_type == TASK_TYPE.CVRPTW:
            return cvrptw_pyvrp(
                task_data=task_data, 
                time_limit=self.time_limit, 
                scale=self.scale, 
                seed=self.seed
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )