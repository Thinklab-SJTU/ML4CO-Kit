r"""
Gurobi Solver for MVC
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
from ml4co_kit.task.graph.mvc import MVCTask


def mvc_gurobi(
    task_data: MVCTask,
    gurobi_time_limit: float = 10.0
):
    # Preparation
    task_data.remove_self_loop()
    nodes_num = task_data.nodes_num
    senders = task_data.edge_index[0]
    receivers = task_data.edge_index[1]
    edge_list = [(min([s, r]), max([s, r])) for s,r in zip(senders, receivers)]
    unique_edge_List = set(edge_list)
    
    # Create gurobi model
    model = gp.Model(f"MVC-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    
    # Constr.
    var_dict = model.addVars(nodes_num, vtype=gp.GRB.BINARY)
    for (s, r) in unique_edge_List:
        xs = var_dict[s]
        xr = var_dict[r]
        model.addConstr(xs + xr >= 1, name="e%d-%d" % (s, r))
    
    # Object
    object = gp.quicksum(
        var_dict[int(n)] * task_data.nodes_weight[n] \
        for n in range(nodes_num)
    )
    model.setObjective(object, gp.GRB.MINIMIZE)

    # Solve
    model.write(f"MVC-{task_data.name}.lp")
    model.optimize()
    os.remove(f"MVC-{task_data.name}.lp")
    
    # Get & Store the solution
    sol = np.array([int(var_dict[key].X) for key in var_dict])
    task_data.from_data(sol=sol, ref=False)