r"""
Gurobi Solver for HCP
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
from ml4co_kit.task.routing.hcp import HCPTask


def hcp_gurobi(
    task_data: HCPTask,
    gurobi_time_limit: float = 60.0,
    gurobi_hcp_use_mtz_or_lazy: str = "lazy",
):
    if gurobi_hcp_use_mtz_or_lazy == "mtz":
        return _hcp_gurobi_mtz(
            task_data=task_data,
            gurobi_time_limit=gurobi_time_limit
        )
    elif gurobi_hcp_use_mtz_or_lazy == "lazy":
        return _hcp_gurobi_lazy(
            task_data=task_data,
            gurobi_time_limit=gurobi_time_limit)
    else:
        raise ValueError(
            f"Invalid value for gurobi_hcp_use_mtz_or_lazy: {gurobi_hcp_use_mtz_or_lazy}"
        )
            

def _hcp_gurobi_mtz(
    task_data: HCPTask,
    gurobi_time_limit: float = 60.0
):
    # Preparation
    nodes_num = task_data.nodes_num
    adj_matrix = task_data.adj_matrix
    
    # Create gurobi model
    model = gp.Model(f"HCP-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    
    # Variables
    x = {}
    for i in range(nodes_num):
        for j in range(nodes_num):
            if i == j:
                continue
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}")
    
    # MTZ variables
    u = {}
    for i in range(nodes_num):
        u[i] = model.addVar(
            lb=0.0, 
            ub=nodes_num-1, 
            vtype=gp.GRB.CONTINUOUS, 
            name=f"u_{i}"
        )
    model.addConstr(u[0] == 0.0)
    model.update()
    
    # Degree constraints
    for i in range(nodes_num):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(nodes_num) if j != i) == 1, 
            name=f"out_{i}"
        )
    for j in range(nodes_num):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(nodes_num) if i != j) == 1, 
            name=f"in_{j}"
        )
    
    # Edge existence constraints
    for i in range(nodes_num):
        for j in range(nodes_num):
            if i != j and adj_matrix[i, j] == 0:
                model.addConstr(x[i, j] == 0, name=f"no_edge_{i}_{j}")
    
    # MTZ subtour elimination
    for i in range(1, nodes_num):
        for j in range(1, nodes_num):
            if i == j:
                continue
            model.addConstr(
                u[i] - u[j] + (nodes_num * x[i, j]) <= nodes_num - 1, 
                name=f"mtz_{i}_{j}"
            )
    
    # Objective
    model.setObjective(0, gp.GRB.MINIMIZE)

    # Solve
    model.write(f"HCP-{task_data.name}.lp")
    model.optimize()
    os.remove(f"HCP-{task_data.name}.lp")
    
    # Get & Store the solution
    try:
        solution_adj_matrix = np.zeros(shape=(nodes_num, nodes_num), dtype=np.int32)
        for i in range(nodes_num):
            for j in range(nodes_num):
                if i == j:
                    continue
                solution_adj_matrix[i][j] = int(round(x[i, j].X))
    except:
        raise Exception("No solution found")
    
    # Get the tour from the adjacency matrix
    tour = _extract_tour_from_adj_matrix(solution_adj_matrix)
    
    # Store the tour in the task_data
    task_data.from_data(adj_matrix=adj_matrix, sol=tour, ref=False)
    
    
def _hcp_gurobi_lazy(
    task_data: HCPTask,
    gurobi_time_limit: float = 60.0
):
    # Preparation
    nodes_num = task_data.nodes_num
    adj_matrix = task_data.adj_matrix
    
    # Create gurobi model
    model = gp.Model(f"HCP-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    model.Params.LazyConstraints = 1
    
    # Variables - only for existing edges
    index_pairs = []
    for i in range(nodes_num):
        for j in range(nodes_num):
            if i != j and adj_matrix[i, j] == 1:
                index_pairs.append((i, j))
    
    x = model.addVars(index_pairs, vtype=gp.GRB.BINARY, name="x")
    
    # Degree constraints
    for i in range(nodes_num):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(nodes_num) if (i, j) in index_pairs) == 1, 
            name=f"out_{i}"
        )
    for j in range(nodes_num):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(nodes_num) if (i, j) in index_pairs) == 1, 
            name=f"in_{j}"
        )
    
    # Objective
    model.setObjective(0, gp.GRB.MINIMIZE)
    
    # Callback functions
    def subtour_edges_from_solution(sol_edges):
        Nloc = nodes_num
        succ = [-1] * Nloc
        for (i, j) in sol_edges:
            succ[i] = j
        visited = [False] * Nloc
        cycles = []
        for start in range(Nloc):
            if visited[start]:
                continue
            cur = start
            cycle = []
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = succ[cur]
                
            if cur in cycle:
                idx = cycle.index(cur)
                cyc = cycle[idx:]
                cycles.append(cyc)
                
        return cycles

    def callback(model: gp.Model, where: int):
        if where == gp.GRB.Callback.MIPSOL:
            sel_edges = []
            for (i, j) in index_pairs:
                val = model.cbGetSolution(x[i, j])
                if val > 0.5:
                    sel_edges.append((i, j))

            cycles = subtour_edges_from_solution(sel_edges)

            for cyc in cycles:
                if len(cyc) == nodes_num:
                    continue
                S = set(cyc)
                expr = gp.quicksum(x[i, j] for i in S for j in S if (i, j) in index_pairs)
                model.cbLazy(expr <= len(S) - 1)

    # Optimize
    model._x = x
    model.optimize(callback)

    # Get & Store the solution
    try:
        solution_adj_matrix = np.zeros(shape=(nodes_num, nodes_num), dtype=np.int32)
        for (i, j) in index_pairs:
            solution_adj_matrix[i][j] = int(round(x[i, j].X))
    except:
        raise Exception("No solution found")

    # Get the tour from the adjacency matrix
    tour = _extract_tour_from_adj_matrix(solution_adj_matrix)
    
    # Store the tour in the task_data
    task_data.from_data(adj_matrix=adj_matrix, sol=tour, ref=False)


def _extract_tour_from_adj_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    n = adj_matrix.shape[0]
    tour = [0]
    cur_node = 0
    cur_idx = 0
    while(len(tour) < adj_matrix.shape[0] + 1):
        cur_idx += 1
        cur_node = np.nonzero(adj_matrix[cur_node])[0]
        if cur_idx == 1:
            cur_node = cur_node.max()
        else:
            cur_node = cur_node[1] if cur_node[0] == tour[-2] else cur_node[0]
        tour.append(cur_node)
    tour = np.array(tour)
    return tour