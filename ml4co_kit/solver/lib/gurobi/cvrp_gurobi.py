r"""
Gurobi Solver for CVRP
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
import gurobipy as gp
from ml4co_kit.task.routing.cvrp import CVRPTask


def cvrp_gurobi(
    task_data: CVRPTask,
    gurobi_time_limit: float = 10.0
):
    """
    Gurobi + Miller-Tucker-Zemlin
    """
    # Preparation
    dists = task_data._get_dists()
    nodes_num = task_data.nodes_num
    norm_demands = task_data.norm_demands
    
    # Create gurobi model
    model = gp.Model(f"CVRP-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    
    # Decision vars: x[i,j] = 1 if arc (i,j) is used
    x = model.addVars(
        range(nodes_num+1), range(nodes_num+1), vtype=gp.GRB.BINARY, name="x"
    )
    
    # MTZ variables for subtour elimination (load / flow formulation)
    u = model.addVars(
        range(1, nodes_num+1), 
        lb={i: norm_demands[i-1] for i in range(1, nodes_num+1)}, 
        ub=1.0, 
        name="u"
    )
    
    # Objective: minimize distance
    model.setObjective(
        expr=gp.quicksum(dists[i][j] * x[i,j] for i in range(nodes_num+1) \
            for j in range(nodes_num+1) if i != j), 
        sense=gp.GRB.MINIMIZE
    )
    
    # Constraints
    # 1. Each customer visited exactly once
    for j in range(1, nodes_num+1):
        model.addConstr(gp.quicksum(x[i,j] for i in range(nodes_num+1) if i != j) == 1)
    for i in range(1, nodes_num+1):
        model.addConstr(gp.quicksum(x[i,j] for j in range(nodes_num+1) if i != j) == 1)

    # 2. Vehicle flow conservation at depot
    model.addConstr(gp.quicksum(x[0,j] for j in range(1, nodes_num+1)) >= 1)
    model.addConstr(gp.quicksum(x[i,0] for i in range(1, nodes_num+1)) >= 1)
    
    # 3. MTZ subtour elimination with capacity
    for i in range(1, nodes_num+1):
        for j in range(1, nodes_num+1):
            if i != j:
                model.addConstr(u[i] - u[j] + 1.0 * x[i,j] <= 1.0 - norm_demands[j-1])
     
    # Optimize
    model.optimize()

    # Get & Store the solution
    try:
        adj_matrix = np.zeros(shape=(nodes_num+1, nodes_num+1), dtype=np.int32)
        for i in range(nodes_num+1):
            for j in range(nodes_num+1):
                if i == j:
                    continue
                adj_matrix[i][j] = int(x[i, j].X)
    except:
        raise Exception("No solution found")

    # Get the tour from the adjacency matrix
    tour = [0]
    cur_node = 0
    visited = np.zeros_like(norm_demands).astype(np.bool_)
    while not visited.all():
        if cur_node == 0:
            next_nodes = np.nonzero(adj_matrix[cur_node])[0]
            next_node = next_nodes[0]
            adj_matrix[0][next_node] = 0
        else:
            next_nodes = np.nonzero(adj_matrix[cur_node])[0]
            next_node = next_nodes[0]
        tour.append(next_node)
        visited[next_node-1] = True
        cur_node = next_node
    tour.append(0)
    tour = np.array(tour)
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)