r"""
OR-Tools Solver for MCl
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


import copy
import numpy as np
from ortools.sat.python import cp_model
from ml4co_kit.task.graph.mcl import MClTask


def mcl_ortools(
    task_data: MClTask, 
    ortools_scale: int = 1e6,
    ortools_time_limit: int = 10
):
    # Preparation 
    cp_task_data = copy.deepcopy(task_data)
    cp_task_data.make_complement()
    cp_task_data.remove_self_loop()
    nodes_num = cp_task_data.nodes_num
    
    # Create OR-Tools Model
    model = cp_model.CpModel()
    
    # Edge List
    senders = cp_task_data.edge_index[0]
    receivers = cp_task_data.edge_index[1]
    edge_list = [(min([s, r]), max([s, r])) for s,r in zip(senders, receivers)]
    unique_edge_list = set(edge_list)
    
    # Vars.
    vertices = np.arange(nodes_num)
    x = {v: model.NewBoolVar(f'x_{v}') for v in vertices}
    
    # Constr.
    for (u, v) in unique_edge_list:
        model.AddBoolOr([x[u].Not(), x[v].Not()])
    
    # Object
    if cp_task_data.node_weighted:
        nodes_weight = (cp_task_data.nodes_weight * ortools_scale).astype(np.int32)
        model.Maximize(sum(x[v] * nodes_weight[v] for v in vertices))
    else:
        model.Maximize(sum(x[v] for v in vertices))
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = ortools_time_limit
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        clique_set = [v for v in vertices if solver.BooleanValue(x[v])]
    else:
        raise ValueError("no feasible solution has been found")

    # solution
    sol = np.zeros(shape=(nodes_num,))
    sol[clique_set] = 1

    # Store the tour in the task_data
    task_data.from_data(sol=sol, ref=False)