r"""
Gurobi Solver for MOPO
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
from ml4co_kit.task.portfolio.mopo import MOPOTask


def mopo_gurobi(
    task_data: MOPOTask,
    gurobi_time_limit: float = 10.0,
):
    # Preparation 
    returns = task_data.returns
    cov = task_data.cov
    var_factor = task_data.var_factor
    ret_factor = task_data.ret_factor
    n_assets = task_data.num_assets
    
    # Create gurobi model
    model = gp.Model("MOPO")
    model.Params.outputFlag = False
    model.Params.timeLimit = gurobi_time_limit
    
    # Create decision variables (portfolio weights)
    w = model.addVars(n_assets, lb=0.0, ub=1.0, name="w")
    
    # Objective: minimize weighted combination of variance and negative returns
    # var_factor * w^T Σ w - ret_factor * r^T w
    variance_expr = gp.QuadExpr()
    for i in range(n_assets):
        for j in range(n_assets):
            variance_expr += w[i] * cov[i, j] * w[j]
    
    returns_expr = gp.quicksum(returns[i] * w[i] for i in range(n_assets))
    
    model.setObjective(
        var_factor * variance_expr - ret_factor * returns_expr,
        gp.GRB.MINIMIZE
    )
    
    # Constraint: weights must sum to 1
    model.addConstr(
        gp.quicksum(w[i] for i in range(n_assets)) == 1.0,
        name="budget_constraint"
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
