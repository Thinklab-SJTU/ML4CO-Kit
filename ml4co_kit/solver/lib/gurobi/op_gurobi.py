r"""
Gurobi Solver for TSP
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


import itertools
import numpy as np
import gurobipy as gp
from ml4co_kit.task.routing.op import OPTask


def op_gurobi(
    task_data: OPTask,
    gurobi_time_limit: float = 10.0,
):
    # Preparation data
    dists = task_data._get_dists()
    prizes = task_data.prizes
    max_length = task_data.max_length
    nodes_num = len(dists)

    # callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model: gp.Model, where: int):
        if where == gp.GRB.Callback.MIPSOL:
            # Make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected_edges = gp.tuplelist((i, j) for (i, j) in model._vars.keys() if vals[i, j] > 0.5)

            # Find all connected components
            unvisited_nodes = list(range(nodes_num))
            # Remove depot from initial unvisited list for subtour detection logic
            # If we find a component containing the depot, it's the main tour.
            if 0 in unvisited_nodes:
                unvisited_nodes.remove(0)

            # Build adjacency list for current solution
            adj = [[] for _ in range(nodes_num)]
            for i, j in selected_edges:
                adj[i].append(j)
                adj[j].append(i)

            # Iterate to find all subtours
            while unvisited_nodes:
                # Start a BFS/DFS from an unvisited node
                start_node = unvisited_nodes[0]
                q = [start_node]
                current_component = set(q)
                head = 0
                while head < len(q):
                    curr = q[head]
                    head += 1
                    for neighbor in adj[curr]:
                        if neighbor not in current_component:
                            current_component.add(neighbor)
                            q.append(neighbor)

                # Remove nodes of this component from unvisited_nodes
                unvisited_nodes = [node for node in unvisited_nodes if node not in current_component]

                # Check if this component is a subtour to eliminate
                if 0 not in current_component: # It's a proper subtour if it doesn't contain the depot
                    # Add subtour elimination constraint: sum of edges within the subtour <= |S| - 1
                    model.cbLazy(
                        gp.quicksum(model._vars[min(i,j), max(i,j)] 
                                    for i, j in itertools.combinations(current_component, 2) 
                                    if (min(i,j), max(i,j)) in model._vars) # Only include existing vars
                        <= len(current_component) - 1
                    )

    # Create Gurobi model
    model = gp.Model()
    model.Params.outputFlag = False

    # Create variables
    x = model.addVars(
        itertools.combinations(range(nodes_num), 2), 
        vtype=gp.GRB.BINARY, name='x'
    )
    
    # Node selection variables (delta_i = 1 if node i is visited)
    prize_dict = {
        i + 1: -p  # We need to maximize so negate
        for i, p in enumerate(prizes)
    }
    
    # delta variables are for nodes 1 to n-1 (since node 0 is depot and always visited)
    delta = model.addVars(
        range(1, nodes_num), 
        obj=prize_dict, 
        vtype=gp.GRB.BINARY, 
        name='delta'
    )
    
    # Degree constraints
    # For depot (node 0): sum of x_0j = 2   
    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, nodes_num)) == 2, 
        name='depot_degree'
    )

    # For other nodes i (1 to n-1): sum of x_ij = 2 * delta_i
    for i in range(1, nodes_num):
        # Sum of edges connected to node i where i < j: x[i,j]
        # Sum of edges connected to node i where k < i: x[k,i]
        model.addConstr(
            gp.quicksum(x[min(i,j), max(i,j)] for j \
                in range(nodes_num) if i != j) == 2 * delta[i], 
            name=f'degree_{i}'
        )

    # Tour length constraint
    model.addConstr(
        gp.quicksum(x[i, j] * dists[i, j] for i, j \
            in itertools.combinations(range(nodes_num), 2)) <= max_length, 
        name='max_length'
    )

    # Set model references for callback
    model._vars = x # Now _vars are the x_ij variables
    model._dvars = delta
    
    # Set parameters
    model.Params.lazyConstraints = 1
    model.Params.threads = 1 # Keep 1 thread for callback
    model.Params.timeLimit = gurobi_time_limit

    # Optimize model
    model.optimize(subtourelim)

    # Extract solution
    if model.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SOLUTION_LIMIT]:
        # Get the values of the edge variables (x_ij where i < j)
        # Using a small tolerance for floating point comparisons to be safe
        vals_x = model.getAttr('x', x)
        selected_edges_raw = [
            (i, j) for (i, j) in x.keys() if vals_x[i, j] > 0.5 - 1e-9
        ] # Added tolerance
        
        # Reconstruct the tour path (starts and ends at depot 0)
        # Build an adjacency list from the selected edges
        adj_solution = [[] for _ in range(nodes_num)] # n is total nodes including depot
        for i, j in selected_edges_raw:
            adj_solution[i].append(j)
            adj_solution[j].append(i) # Add both directions for traversal

    # Start from depot (node 0)
    reconstructed_path = []
    current_node = 0
    reconstructed_path.append(current_node)
    
    # The first step from the depot (choose one of its two neighbors)
    # We need to know the previous node to avoid immediately going back.
    if adj_solution[0]: # Check if there are any edges from depot
        prev_node = 0 # Initialize previous node as depot itself
        current_node = adj_solution[0][0] # Take the first neighbor
        reconstructed_path.append(current_node)
        
    # Loop until we return to the depot (node 0)
    path_complete = False
    while current_node != 0:
        found_next_step = False
        for neighbor in adj_solution[current_node]:
            # If we find the depot and the path is long enough (more than just 0 -> neighbor -> 0)
            if neighbor == 0 and len(reconstructed_path) > 2:
                reconstructed_path.append(0)
                path_complete = True
                found_next_step = True
                break # Tour completed
            # If the neighbor is not the previous node and hasn't 
            # been visited yet (for non-depot nodes)
            elif neighbor != prev_node: # Avoid immediate backtracking
                # Ensure we don't visit nodes twice (except the depot at the end)
                if neighbor not in reconstructed_path: 
                    reconstructed_path.append(neighbor)
                    prev_node = current_node
                    current_node = neighbor 
                    found_next_step = True
                    break
        
        if path_complete:
            break

        if not found_next_step:
            msg = (
                f"Error: Could not complete path reconstruction for instance. "
                f"Stuck at node {current_node}. Adjacency: {adj_solution[current_node]}. "
                f"Current path: {reconstructed_path}"
            )
            raise ValueError(msg)
    
    # Store the tour in the task_data
    task_data.from_data(sol=np.array(reconstructed_path), ref=False)