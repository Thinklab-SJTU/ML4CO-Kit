r"""
KaMIS (Karlsruhe Maximum Independent Sets)
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
from .lib.kamis.mis_kamis import mis_kamis


class KaMISSolver(SolverBase):
    """
    KaMIS: https://github.com/KarlsruheMIS/KaMIS
    CHSZLabLib: https://github.com/CHSZLab/CHSZLabLib
    Current Version: v0.5.27
    Last Update: 2026-05-28
    @article{
        lamm2017finding,
        author  = {Sebastian Lamm and Peter Sanders and Christian Schulz and Darren Strash and Renato F. Werneck},
        title   = {Finding Near-Optimal Independent Sets at Scale},
        journal = {Journal of Heuristics},
        volume  = {23},
        number  = {4},
        pages   = {207--229},
        year    = {2017},
        doi     = {10.1007/s10732-017-9337-x}
    }
    @article{
        hespe2019scalable,
        author  = {Demian Hespe and Christian Schulz and Darren Strash},
        title   = {Scalable Kernelization for Maximum Independent Sets},
        journal = {ACM Journal of Experimental Algorithmics},
        volume  = {24},
        number  = {1},
        pages   = {1.16:1--1.16:22},
        year    = {2019},
        doi     = {10.1145/3355502}
    }
    @inproceedings{
        lamm2019exactly,
        author    = {Sebastian Lamm and Christian Schulz and Darren Strash and Robert Williger and Huashuo Zhang},
        title     = {Exactly Solving the Maximum Weight Independent Set Problem on Large Real-World Graphs},
        booktitle = {Proceedings of the 21st Workshop on Algorithm Engineering and Experiments ({ALENEX})},
        pages     = {144--158},
        publisher = {SIAM},
        year      = {2019},
        doi       = {10.1137/1.9781611975499.12}
    }
    @inproceedings{
        grossmann2023mmwis,
        author    = {Ernestine Gro{\ss}mann and Sebastian Lamm and Christian Schulz and Darren Strash},
        title     = {Finding Near-Optimal Weight Independent Sets at Scale},
        booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference ({GECCO})},
        pages     = {293--302},
        publisher = {ACM},
        year      = {2023},
        doi       = {10.1145/3583131.3590353}
    }
    """
    def __init__(
        self, 
        kamis_time_limit: float = 10.0,
        unweighted_solver_type: str = "redumis", # Support "redumis" and "online_mis"
        weighted_solver_type: str = "mmwis", # Support "mmwis" and "branch_reduce"
        weighted_scale: float = 1e5,
        optimizer: OptimizerBase = None,
        kamis_seed: int = 1234
    ):
        # Super Initialization
        super(KaMISSolver, self).__init__(
            solver_type=SOLVER_TYPE.KAMIS, optimizer=optimizer
        )

        # Set Attributes
        self.kamis_time_limit = kamis_time_limit
        self.unweighted_solver_type = unweighted_solver_type
        self.weighted_solver_type = weighted_solver_type
        self.weighted_scale = weighted_scale
        self.kamis_seed = kamis_seed

        # Check Attributes
        if unweighted_solver_type not in ["redumis", "online_mis"]:
            raise ValueError(f"Invalid unweighted solver type: {unweighted_solver_type}")
        if weighted_solver_type not in ["mmwis", "branch_reduce"]:
            raise ValueError(f"Invalid weighted solver type: {weighted_solver_type}")

    def _solve(self, task_data: TaskBase):
        """Solve the task data using KaMIS Solver."""
        if task_data.task_type == TASK_TYPE.MIS:
            return mis_kamis(
                task_data=task_data, 
                kamis_time_limit=self.kamis_time_limit,
                unweighted_solver_type=self.unweighted_solver_type,
                weighted_solver_type=self.weighted_solver_type,
                weighted_scale=self.weighted_scale,
                kamis_seed=self.kamis_seed
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )