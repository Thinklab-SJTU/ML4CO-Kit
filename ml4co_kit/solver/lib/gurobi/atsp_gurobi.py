r"""
Gurobi Solver for ATSP
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
from ml4co_kit.task.routing.atsp import ATSPTask


def atsp_gurobi(
    task_data: ATSPTask,
    gurobi_time_limit: float = 10.0
):
    """
    Gurobi + lazy subtour elimination callback
    """
    # Preparation
    dists = task_data.dists
    nodes_num = task_data.nodes_num
    
    # Create gurobi model
    model = gp.Model(f"ATSP-{task_data.name}")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)
    model.Params.LazyConstraints = 1
    
    # Variables
    index_pairs = [(i, j) for i in range(nodes_num) for j in range(nodes_num) if i != j]
    x = model.addVars(index_pairs, vtype=gp.GRB.BINARY, name="x")
    
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
    
    # Objetive
    model.setObjective(
        expr=gp.quicksum(dists[i, j] * x[i, j] for (i, j) in index_pairs), 
        sense=gp.GRB.MINIMIZE
    )
    
    # Callback functions
    def subtour_edges_from_solution(sol_edges):
        """
        Given a list of edges (i->j) of a solution, 
        return all sub-tours (each sub-tour is a list of node indices).
        
        Assume that each node has exactly one outgoing edge and 
        one incoming edge (integer solution).
        """
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
                
            # Find the start of a cycle: cycle may contain parts that have already closed
            # Extract the entire closed loop: from the first occurrence of cur to the end
            if cur in cycle:
                idx = cycle.index(cur)
                cyc = cycle[idx:]
                cycles.append(cyc)
                
        return cycles

    def callback(model: gp.Model, where: int):
        if where == gp.GRB.Callback.MIPSOL:
            # Called when an integer feasible solution is found, add subtour constraints as needed
            # Collect the edges selected in the current solution (value > 0.5)
            sel_edges = []
            for (i, j) in index_pairs:
                val = model.cbGetSolution(x[i, j])
                # Because there may be numerical errors, use a threshold of 0.5
                if val > 0.5:
                    sel_edges.append((i, j))

            # Find all sub-tours
            cycles = subtour_edges_from_solution(sel_edges)

            # For each sub-tour, if it does not contain all points, add lazy constraint
            for cyc in cycles:
                if len(cyc) == nodes_num:
                    continue  # Complete tour, no need to add constraint
                S = set(cyc)
                # The total number of edges in the sub-tour <= |S| - 1
                expr = gp.quicksum(x[i, j] for i in S for j in S if i != j)
                # Equivalent SEC: sum_{i in S, j not in S} x[i,j] >= 1
                # Use cbLazy to add constraint
                model.cbLazy(expr <= len(S) - 1)

    # Optimize
    model._x = x
    model.optimize(callback)

    # Get & Store the solution
    try:
        adj_matrix = np.zeros(shape=(nodes_num, nodes_num), dtype=np.int32)
        for (i, j) in index_pairs:
            adj_matrix[i][j] = int(x[i, j].X)
    except:
        raise Exception("No solution found")

    # Get the tour from the adjacency matrix
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
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)