r"""
Gurobi Solver for UNSAT-C
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


import os
import numpy as np
import gurobipy as gp
from ml4co_kit.task.sat.unsatc import USATCTask


def unsatc_gurobi(
    task_data: USATCTask,
    gurobi_time_limit: float = 10.0
):
    model = gp.Model(f"UNSAT-C-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    
    var_dict = model.addVars(task_data.vars_num, vtype=gp.GRB.BINARY)
    clause_vars = model.addVars(len(task_data.clauses), vtype=gp.GRB.BINARY)
    
    for idx, clause in enumerate(task_data.clauses):
        clause_expr = gp.quicksum(
            var_dict[abs(lit) - 1] if lit > 0 else (1 - var_dict[abs(lit) - 1])
            for lit in clause
        )
        model.addConstr(clause_expr >= clause_vars[idx])
    
    model.addConstr(gp.quicksum(clause_vars[i] for i in range(len(task_data.clauses))) < len(task_data.clauses))
    
    var_involvement = model.addVars(task_data.vars_num, vtype=gp.GRB.BINARY)
    for var_idx in range(task_data.vars_num):
        involved_clauses = [
            idx for idx, clause in enumerate(task_data.clauses)
            if any(abs(lit) - 1 == var_idx for lit in clause)
        ]
        if involved_clauses:
            model.addConstr(
                var_involvement[var_idx] >= gp.quicksum(
                    1 - clause_vars[idx] for idx in involved_clauses
                ) / len(involved_clauses)
            )
    
    model.setObjective(
        gp.quicksum(var_involvement[i] for i in range(task_data.vars_num)),
        gp.GRB.MAXIMIZE
    )
    
    model.write(f"UNSAT-C-{task_data.name}.lp")
    model.optimize()
    os.remove(f"UNSAT-C-{task_data.name}.lp")
    
    sol = np.array([int(var_involvement[i].X > 0.5) for i in range(task_data.vars_num)], dtype=bool)
    task_data.sol = sol
