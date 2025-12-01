r"""
Gurobi Solver for MinVarPO
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
import gurobipy as gp
from ml4co_kit.task.portfolio.minvarpo import MinVarPOTask


def minvarpo_gurobi(
    task_data: MinVarPOTask,
    gurobi_time_limit: float = 10.0,
):
    # Preparation 
    returns = task_data.returns
    cov = task_data.cov
    required_returns = task_data.required_returns
    n_assets = task_data.num_assets
    
    # Create gurobi model
    model = gp.Model("MinVarPO")
    model.Params.outputFlag = False
    model.Params.timeLimit = gurobi_time_limit
    
    # Create decision variables (portfolio weights)
    w = model.addVars(n_assets, lb=0.0, ub=1.0, name="w")
    
    # Objective: minimize portfolio variance
    # w^T Σ w
    variance_expr = gp.QuadExpr()
    for i in range(n_assets):
        for j in range(n_assets):
            variance_expr += w[i] * cov[i, j] * w[j]
    
    model.setObjective(variance_expr, gp.GRB.MINIMIZE)
    
    # Constraint 1: weights must sum to 1
    model.addConstr(
        gp.quicksum(w[i] for i in range(n_assets)) == 1.0,
        name="budget_constraint"
    )
    
    # Constraint 2: portfolio returns must meet minimum requirement
    model.addConstr(
        gp.quicksum(returns[i] * w[i] for i in range(n_assets)) >= required_returns,
        name="return_constraint"
    )
    
    # Optimize model
    model.optimize()
    
    # Extract solution
    try:
        solution = np.array([w[i].x for i in range(n_assets)])
        task_data.from_data(sol=solution, ref=False)
    except:
        # If no solution found, raise error
        raise ValueError("No solution found")
