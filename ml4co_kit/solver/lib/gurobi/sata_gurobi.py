r"""
Gurobi Solver for SAT-A
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
from ml4co_kit.task.sat.sata import SATATask


def sata_gurobi(
    task_data: SATATask,
    gurobi_time_limit: float = 10.0
):
    model = gp.Model(f"SAT-A-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    
    var_dict = model.addVars(task_data.vars_num, vtype=gp.GRB.BINARY)
    
    for clause in task_data.clauses:
        clause_expr = gp.quicksum(
            var_dict[abs(lit) - 1] if lit > 0 else (1 - var_dict[abs(lit) - 1])
            for lit in clause
        )
        model.addConstr(clause_expr >= 1)
    
    model.setObjective(0, gp.GRB.MINIMIZE)
    
    model.write(f"SAT-A-{task_data.name}.lp")
    model.optimize()
    os.remove(f"SAT-A-{task_data.name}.lp")
    
    if model.status == gp.GRB.OPTIMAL:
        sol = np.array([int(var_dict[i].X) for i in range(task_data.vars_num)], dtype=bool)
    else:
        sol = np.zeros(task_data.vars_num, dtype=bool)
    
    task_data.sol = sol
