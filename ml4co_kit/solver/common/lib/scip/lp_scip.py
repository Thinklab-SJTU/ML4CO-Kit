r"""
SCIP Solver for LP
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
import scipy.sparse
from pyscipopt import Model, quicksum
from ml4co_kit.task.milp.lp import LPTask


def _row_linear_expr(A_csr: scipy.sparse.csr_matrix, x: dict, row: int):
    """Build ``quicksum`` for one CSR row of the constraint matrix."""
    start: int = A_csr.indptr[row]
    end: int = A_csr.indptr[row + 1]
    return quicksum(
        float(A_csr.data[k]) * x[A_csr.indices[k]] for k in range(start, end)
    )


def lp_scip(
    task_data: LPTask,
    scip_time_limit: float = 10.0,
):
    # Get Data
    vars_num = task_data.vars_num
    constrs_num = task_data.constrs_num
    A_csr = task_data.A
    threshold = task_data.threshold

    # Create SCIP model
    model = Model(task_data.name)
    model.hideOutput()
    model.setRealParam("limits/time", scip_time_limit)

    # Decision variables: x[j] with bounds and type (C / B / I)
    x = {}
    for j in range(vars_num):
        lb = float(task_data.lx[j]) if np.isfinite(task_data.lx[j]) else None
        ub = float(task_data.ux[j]) if np.isfinite(task_data.ux[j]) else None
        x[j] = model.addVar(lb=lb, ub=ub, vtype="C", name=f"x{j}")

    # Objective: minimize c^T x (c is in internal minimization form)
    model.setObjective(
        quicksum(float(task_data.c[j]) * x[j] for j in range(vars_num)),
        "minimize",
    )

    # Constraints: one row of A per iteration; split into =, <=, or >=
    for i in range(constrs_num):
        expr = _row_linear_expr(A_csr, x, i)
        ls_i, us_i = task_data.ls[i], task_data.us[i]
        finite_ls = np.isfinite(ls_i)
        finite_us = np.isfinite(us_i)
        if finite_ls and finite_us and abs(ls_i - us_i) <= threshold:
            model.addCons(expr == float(us_i), name=f"c{i}")
        else:
            if finite_us:
                model.addCons(expr <= float(us_i), name=f"c{i}_le")
            if finite_ls:
                model.addCons(expr >= float(ls_i), name=f"c{i}_ge")

    # Solve and extract the best primal solution if available
    model.optimize()

    # Extract primal solution; raise if none or not feasible
    status = model.getStatus()
    sol = model.getBestSol()
    if sol is None:
        raise ValueError(
            f"SCIP found no primal solution (status={status!r}, "
            f"time_limit={scip_time_limit}s)."
        )

    # Extract solution
    solution = np.array(
        [model.getVal(x[j]) for j in range(vars_num)],
        dtype=task_data.precision,
    )

    # Check constraints
    if not task_data.check_constraints(solution):
        raise ValueError(
            f"SCIP returned a primal that violates task constraints "
            f"(status={status!r}, time_limit={scip_time_limit}s). "
            f"Try increasing ``scip_time_limit``."
        )

    # Set solution
    task_data.from_data(sol=solution, ref=False)
