r"""
SCIP Solver for MaxRetPO
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
from pyscipopt import Model, quicksum
from ml4co_kit.task.portfolio.maxretpo import MaxRetPOTask


def maxretpo_scip(
    task_data: MaxRetPOTask,
    scip_time_limit: float = 10.0,
):
    # Get problem data
    returns = task_data.returns
    cov = task_data.cov
    max_var = task_data.max_var
    n_assets = task_data.num_assets
    
    # Create SCIP model
    model = Model("MaxRetPO")
    model.hideOutput()
    model.setRealParam("limits/time", scip_time_limit)
    
    # Create decision variables (portfolio weights)
    w = {}
    for i in range(n_assets):
        w[i] = model.addVar(lb=0.0, ub=1.0, name=f"w_{i}", vtype="C")
    
    # Objective: maximize portfolio returns (SCIP minimizes by default, so negate)
    model.setObjective(
        -quicksum(float(returns[i]) * w[i] for i in range(n_assets)),
        "minimize"
    )
    
    # Constraint 1: weights must sum to 1
    model.addCons(
        quicksum(w[i] for i in range(n_assets)) == 1.0,
        name="budget_constraint"
    )
    
    # Constraint 2: portfolio variance must not exceed max_var
    # w^T Σ w <= max_var using the new API
    # Build quadratic expression
    quad_expr = quicksum(
        float(cov[i, j]) * w[i] * w[j] 
        for i in range(n_assets) 
        for j in range(n_assets)
    )
    model.addCons(quad_expr <= float(max_var), name="variance_constraint")
    
    # Optimize model
    model.optimize()
    
    # Extract solution
    sol = model.getBestSol()
    if sol is not None:
        solution = np.array([model.getVal(w[i]) for i in range(n_assets)])
        task_data.from_data(sol=solution, ref=False)
    else:
        # If no solution found, return equal weights as fallback
        solution = np.ones(n_assets) / n_assets
        task_data.from_data(sol=solution, ref=False)
