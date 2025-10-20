r"""
Gurobi Solver for Maximum Return Portfolio Optimization (MaxRetPO)
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
from ml4co_kit.task.portfolio.maxret_po import MaxRetPOTask


def maxretpo_gurobi(
    task_data: MaxRetPOTask,
    gurobi_time_limit: float = 10.0,
):
    """
    Solve Maximum Return Portfolio Optimization using Gurobi.
    
    The problem is:
    maximize: r^T w
    subject to:
        sum(w) = 1
        w^T Σ w <= max_var
        w >= 0
    
    where:
    - r is the expected returns vector
    - w is the portfolio weights vector
    - Σ is the covariance matrix
    - max_var is the maximum allowed variance
    """
    # Get problem data
    returns = task_data.returns
    cov = task_data.cov
    max_var = task_data.max_var
    n_assets = task_data.num_assets
    
    # Create Gurobi model
    model = gp.Model("MaxRetPO")
    model.Params.outputFlag = False
    model.Params.timeLimit = gurobi_time_limit
    
    # Create decision variables (portfolio weights)
    w = model.addVars(n_assets, lb=0.0, ub=1.0, name="w")
    
    # Objective: maximize portfolio returns
    model.setObjective(
        gp.quicksum(returns[i] * w[i] for i in range(n_assets)),
        gp.GRB.MAXIMIZE
    )
    
    # Constraint 1: weights must sum to 1
    model.addConstr(
        gp.quicksum(w[i] for i in range(n_assets)) == 1.0,
        name="budget_constraint"
    )
    
    # Constraint 2: portfolio variance must not exceed max_var
    # w^T Σ w <= max_var
    variance_expr = gp.QuadExpr()
    for i in range(n_assets):
        for j in range(n_assets):
            variance_expr += w[i] * cov[i, j] * w[j]
    
    model.addConstr(variance_expr <= max_var, name="variance_constraint")
    
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
