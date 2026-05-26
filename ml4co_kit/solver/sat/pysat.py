r"""
PySAT Solver.
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
from .lib.pysat.satp_pysat import satp_pysat
from .lib.pysat.sata_pysat import sata_pysat


class PySATSolver(SolverBase):
    """
    PySAT: https://github.com/pysathq/pysat
    Current Version: 1.9.dev4
    Last Update: 2026-05-26
    @inproceedings{
        ignatiev2018pysat,
        title={PySAT: A Python toolkit for prototyping with SAT oracles},
        author={Ignatiev, Alexey and Morgado, Antonio and Marques-Silva, Joao},
        booktitle={International Conference on Theory and Applications of Satisfiability Testing},
        pages={428--437},
        year={2018},
        organization={Springer}
    }
    @inproceedings{
        ignatiev2024towards,
        title={Towards universally accessible SAT technology},
        author={Ignatiev, Alexey and Tan, Zi Li and Karamanos, Christos},
        booktitle={27th International Conference on Theory and Applications of Satisfiability Testing (SAT 2024)},
        pages={16--1},
        year={2024},
        organization={Schloss Dagstuhl--Leibniz-Zentrum f{\"u}r Informatik}
    }
    """
    def __init__(
        self,
        pysat_solver_name: str = "cadical195",
        pysat_solver_args: dict = {},
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(PySATSolver, self).__init__(
            solver_type=SOLVER_TYPE.PYSAT, optimizer=optimizer
        )

        # Get solver from ``pysat``
        self.solver_name = pysat_solver_name
        self.solver_args = pysat_solver_args

    def _solve(self, task_data: TaskBase):
        """Solve the task data using PySAT Solver."""
        if task_data.task_type == TASK_TYPE.SATP:
            return satp_pysat(
                task_data=task_data, 
                solver_name=self.solver_name,
                solver_args=self.solver_args
            )
        elif task_data.task_type == TASK_TYPE.SATA:
            return sata_pysat(
                task_data=task_data, 
                solver_name=self.solver_name,
                solver_args=self.solver_args
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )