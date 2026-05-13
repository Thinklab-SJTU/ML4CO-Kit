r"""
PySAT Solver for SAT-A
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


import numpy as np
import pysat.solvers
from ml4co_kit.task.sat.sata import SATATask


def sata_pysat(
    task_data: SATATask, solver_name: str, solver_args: dict
):
    # Create solver
    solver = pysat.solvers.Solver(solver_name, **solver_args)

    # Add clauses to solver
    for clause in task_data.clauses:
        solver.add_clause(clause)
    
    # Solve the problem
    if solver.solve():
        idx_sol = np.array(solver.get_model())
        bool_sol = np.zeros(len(idx_sol), dtype=np.bool_)
        bool_sol[idx_sol > 0] = True
        bool_sol[idx_sol < 0] = False
    else:
        raise ValueError("Failed to solve the problem!")

    # Store the solution
    task_data.from_data(sol=bool_sol, ref=False)