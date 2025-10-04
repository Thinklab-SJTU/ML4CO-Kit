r"""
Gurobi Solver for MCut
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
from ml4co_kit.task.graph.mcut import MCutTask


def mcut_gurobi(
    task_data: MCutTask,
    gurobi_time_limit: float = 10.0
):
    # Preparation 
    nodes_num = task_data.nodes_num
    senders = task_data.edge_index[0]
    receivers = task_data.edge_index[1]
    edge_attr = task_data.edges_weight
    
    # Create gurobi model
    model = gp.Model(f"MCut-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    
    # Object
    var_dict = model.addVars(nodes_num, vtype=gp.GRB.BINARY)
    object = gp.quicksum( 
        (2 * var_dict[int(s)] - 1) * weight * (2 * var_dict[int(r)] - 1) / 2 
        for s, r, weight in zip(senders, receivers, edge_attr)
    )
    model.setObjective(object, gp.GRB.MINIMIZE)

    # Solve
    model.write(f"MCut-{task_data.name}.lp")
    model.optimize()
    os.remove(f"MCut-{task_data.name}.lp")
    
    # Get & Store the solution
    sol = np.array([int(var_dict[key].X) for key in var_dict])
    task_data.from_data(sol=sol, ref=False)