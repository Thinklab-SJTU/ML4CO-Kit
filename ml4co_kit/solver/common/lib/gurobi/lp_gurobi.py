r"""
Gurobi Solver for LP
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
import gurobipy as gp
from ml4co_kit.task.milp.lp import LPTask


def lp_gurobi(
    task_data: LPTask,
    gurobi_time_limit: float = 10.0
):
    # Create Gurobi environment
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    # Build Gurobi model from task data
    model = task_data._build_gurobi_model(env)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)

    # Solve
    model.optimize()

    # Extract primal solution; raise if none or not feasible
    status = model.Status
    if model.SolCount == 0:
        raise ValueError(
            f"Gurobi found no primal solution (status={status!r}, "
            f"time_limit={gurobi_time_limit}s)."
        )

    # Extract primal solution
    solution = np.array(
        [var.X for var in model.getVars()],
        dtype=task_data.precision,
    )

    # Check if the solution is feasible
    if not task_data.check_constraints(solution):
        raise ValueError(
            f"Gurobi returned a primal that violates task constraints "
            f"(status={status!r}, time_limit={gurobi_time_limit}s). "
            f"Try increasing ``gurobi_time_limit``."
        )

    # Store solution
    task_data.from_data(sol=solution, ref=False)

    # Dispose Gurobi model and environment
    model.dispose()
    env.dispose()
