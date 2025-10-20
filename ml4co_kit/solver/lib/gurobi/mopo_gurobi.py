r"""
Gurobi Solver for Multi-Objective Portfolio Optimization (MOPO)
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
from ml4co_kit.task.portfolio.mo_po import MOPOTask


def mopo_gurobi(
    task_data: MOPOTask,
    gurobi_time_limit: float = 10.0,
):
    """
    Solve Multi-Objective Portfolio Optimization using Gurobi.
    
    The problem is:
    minimize: var_factor * w^T Σ w - ret_factor * r^T w
    subject to:
        sum(w) = 1
        w >= 0
    
    where:
    - r is the expected returns vector
    - w is the portfolio weights vector
    - Σ is the covariance matrix
    - var_factor is the weight for variance term
    - ret_factor is the weight for return term (ret_factor = 1 - var_factor)
    """
    # Get problem data
    returns = task_data.returns
    cov = task_data.cov
    var_factor = task_data.var_factor
    ret_factor = task_data.ret_factor
    n_assets = task_data.num_assets
    
    # Create Gurobi model
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
    if model.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SOLUTION_LIMIT]:
        solution = np.array([w[i].x for i in range(n_assets)])
        task_data.from_data(sol=solution, ref=False)
    else:
        # If no solution found, return equal weights as fallback
        solution = np.ones(n_assets) / n_assets
        task_data.from_data(sol=solution, ref=False)
