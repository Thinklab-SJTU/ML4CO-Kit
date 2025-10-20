r"""
SCIP Solver for MinVarPO
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
from ml4co_kit.task.portfolio.minvarpo import MinVarPOTask


def minvarpo_scip(
    task_data: MinVarPOTask,
    scip_time_limit: float = 10.0,
):
    # Get problem data
    returns = task_data.returns
    cov = task_data.cov
    required_returns = task_data.required_returns
    n_assets = task_data.num_assets
    
    # Create SCIP model
    model = Model("MinVarPO")
    model.hideOutput()
    model.setRealParam("limits/time", scip_time_limit)
    
    # Create decision variables (portfolio weights)
    w = {}
    for i in range(n_assets):
        w[i] = model.addVar(lb=0.0, ub=1.0, name=f"w_{i}", vtype="C")
    
    # Create auxiliary variable for the objective (portfolio variance)
    # We minimize t subject to w^T Σ w <= t
    t = model.addVar(lb=0.0, name="variance", vtype="C")
    
    # Objective: minimize portfolio variance (represented by auxiliary variable t)
    model.setObjective(t, "minimize")
    
    # Constraint 1: weights must sum to 1
    model.addCons(
        quicksum(w[i] for i in range(n_assets)) == 1.0,
        name="budget_constraint"
    )
    
    # Constraint 2: portfolio returns must meet minimum requirement
    model.addCons(
        quicksum(float(returns[i]) * w[i] for i in range(n_assets)) >= float(required_returns),
        name="return_constraint"
    )
    
    # Constraint 3: w^T Σ w <= t (quadratic constraint)
    # Build quadratic expression using the new API
    quad_expr = quicksum(
        float(cov[i, j]) * w[i] * w[j] 
        for i in range(n_assets) 
        for j in range(n_assets)
    )
    model.addCons(quad_expr <= t, name="variance_bound")
    
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
