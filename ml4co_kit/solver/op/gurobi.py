r"""
Gurobi Solver for solving OP.

Gurobi is a versatile optimization solver used for solving various types of
mathematical programming problems, including linear programming (LP), 
mixed-integer programming (MIP), quadratic programming (QP), and more.
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
import math
import uuid
import itertools
import numpy as np
import gurobipy as gp
from typing import Union, Tuple, List, Optional
from multiprocessing import Pool
from ml4co_kit.solver.op.base import OPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer

MAX_LENGTH_TOL = 1e-5  # Tolerance for maximum length check

class OPGurobiSolver(OPSolver):
    def __init__(self, time_limit: float = 60.0):
        super(OPGurobiSolver, self).__init__(
            solver_type=SOLVER_TYPE.GUROBI, time_limit=time_limit
        )
        self.threads = None
        self.gap = None
        
    def _solve_single_instance(
        self, 
        depot: np.ndarray, 
        loc: np.ndarray, 
        prize: np.ndarray, 
        max_length: float
    ) -> Tuple[float, List[int]]:
        """
        Solve a single OP instance using Gurobi
        
        :param depot: Depot coordinates (2,)
        :param loc: Node coordinates (n, 2)
        :param prize: Node prizes (n,)
        :param max_length: Maximum tour length
        :return: (objective value, tour)
        """
        depot_np = np.array(depot)
        loc_np = np.array(loc)
        
        points_np = np.vstack((depot_np[None, :], loc_np))
        n = len(points_np)

        # Callback - use lazy constraints to eliminate sub-tours
        def subtourelim(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                # Make a list of edges selected in the solution
                vals = model.cbGetSolution(model._vars)
                selected_edges = gp.tuplelist((i, j) for (i, j) in model._vars.keys() if vals[i, j] > 0.5)

                # Find all connected components
                unvisited_nodes = list(range(n))
                # Remove depot from initial unvisited list for subtour detection logic
                # If we find a component containing the depot, it's the main tour.
                if 0 in unvisited_nodes:
                    unvisited_nodes.remove(0)

                # Build adjacency list for current solution
                adj = [[] for _ in range(n)]
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

        # Given a tuplelist of selected edges (i,j) where i < j, find the shortest subtour
        def subtour(edges, n_nodes, exclude_depot=True):
            # Build an adjacency list from the selected edges
            adj = [[] for _ in range(n_nodes)]
            for i, j in edges:
                adj[i].append(j)
                adj[j].append(i)

            unvisited = list(range(n_nodes))
            cycle = None

            while unvisited:
                thiscycle = []
                q = [unvisited[0]] # Start BFS/DFS from first unvisited node
                visited_in_component = set(q)

                head = 0
                while head < len(q):
                    curr = q[head]
                    head += 1
                    thiscycle.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited_in_component:
                            visited_in_component.add(neighbor)
                            q.append(neighbor)

                # Update unvisited
                for node in thiscycle:
                    if node in unvisited:
                        unvisited.remove(node)

                # Check if this component is a proper subtour
                if (
                    len(thiscycle) > 1 and                           # Must be more than one node
                    not (0 in thiscycle and exclude_depot) and      # If depot is in, it's not a "subtour" to eliminate (unless it's not the full tour)
                    (cycle is None or len(cycle) > len(thiscycle))  # Keep the shortest one
                ):
                    cycle = thiscycle
            return cycle

        # Calculate Euclidean distances between each pair of points
        dist = {(i,j) : np.linalg.norm(points_np[i] - points_np[j])
                for i, j in itertools.combinations(range(n), 2)}

        # Create Gurobi model
        model = gp.Model()
        model.Params.outputFlag = False

        # Create variables
        x = model.addVars(itertools.combinations(range(n), 2), vtype=gp.GRB.BINARY, name='x')
        
        # Node selection variables (delta_i = 1 if node i is visited)
        prize_dict = {
            i + 1: -p  # We need to maximize so negate
            for i, p in enumerate(prize)
        }
        
        # delta variables are for nodes 1 to n-1 (since node 0 is depot and always visited)
        delta = model.addVars(range(1, n), obj=prize_dict, vtype=gp.GRB.BINARY, name='delta')
        
        # Degree constraints
        # For depot (node 0): sum of x_0j = 2   
        model.addConstr(gp.quicksum(x[0, j] for j in range(1, n)) == 2, name='depot_degree')

        # For other nodes i (1 to n-1): sum of x_ij = 2 * delta_i
        for i in range(1, n):
            # Sum of edges connected to node i where i < j: x[i,j]
            # Sum of edges connected to node i where k < i: x[k,i]
            model.addConstr(gp.quicksum(x[min(i,j), max(i,j)] for j in range(n) if i != j) == 2 * delta[i], name=f'degree_{i}')

        # Tour length constraint
        model.addConstr(gp.quicksum(x[i, j] * dist[i, j] for i, j in itertools.combinations(range(n), 2)) <= max_length, name='max_length')

        # Set model references for callback
        model._vars = x # Now _vars are the x_ij variables
        model._dvars = delta
        
        # Set parameters
        model.Params.lazyConstraints = 1
        model.Params.threads = 1 # Keep 1 thread for callback
        if self.time_limit:
            model.Params.timeLimit = self.time_limit
        if self.gap:
            model.Params.mipGap = self.gap * 0.01  # Percentage

        # Optimize model
        model.optimize(subtourelim)

        # Extract solution
        if model.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SOLUTION_LIMIT]:
            try:
                # Get the values of the edge variables (x_ij where i < j)
                # Using a small tolerance for floating point comparisons to be safe
                vals_x = model.getAttr('x', x)
                selected_edges_raw = [(i, j) for (i, j) in x.keys() if vals_x[i, j] > 0.5 - 1e-9] # Added tolerance
                
                # Get the values of the delta variables (node selection)
                vals_delta = model.getAttr('x', delta)
                selected_nodes_indices_gurobi = [k for k in delta.keys() if vals_delta[k] > 0.5 - 1e-9] # Added tolerance
                
                # Reconstruct the tour path (starts and ends at depot 0)
                # Build an adjacency list from the selected edges
                adj_solution = [[] for _ in range(n)] # n is total nodes including depot
                for i, j in selected_edges_raw:
                    adj_solution[i].append(j)
                    adj_solution[j].append(i) # Add both directions for traversal

                reconstructed_path = []
                
                # Verify depot's degree
                if len(adj_solution[0]) != 2:
                    # This indicates a problem with the Gurobi solution if it's supposed to be a tour.
                    # It implies the degree constraint on depot (sum(x_0j) == 2) was violated or not fully enforced.
                    print(f"Warning: Depot (node 0) does not have degree 2 in the extracted solution. Actual degree: {len(adj_solution[0])}")
                    # Attempt to find *any* path if degree is not 2, or return default
                    return 0.0, [0] 

                # Start from depot (node 0)
                current_node = 0
                reconstructed_path.append(current_node)
                
                # The first step from the depot (choose one of its two neighbors)
                # We need to know the previous node to avoid immediately going back.
                if adj_solution[0]: # Check if there are any edges from depot
                    prev_node = 0 # Initialize previous node as depot itself
                    current_node = adj_solution[0][0] # Take the first neighbor
                    reconstructed_path.append(current_node)
                else:
                    # If depot has no edges, it means no path was found (or only depot is selected)
                    print("Warning: Depot has no outgoing edges in the solution.")
                    return 0.0, [0]

                path_complete = False
                # Loop until we return to the depot (node 0)
                while current_node != 0:
                    found_next_step = False
                    for neighbor in adj_solution[current_node]:
                        # If we find the depot and the path is long enough (more than just 0 -> neighbor -> 0)
                        if neighbor == 0 and len(reconstructed_path) > 2:
                            reconstructed_path.append(0)
                            path_complete = True
                            found_next_step = True
                            break # Tour completed
                        # If the neighbor is not the previous node and hasn't been visited yet (for non-depot nodes)
                        elif neighbor != prev_node: # Avoid immediate backtracking
                            if neighbor not in reconstructed_path: # Ensure we don't visit nodes twice (except the depot at the end)
                                reconstructed_path.append(neighbor)
                                prev_node = current_node
                                current_node = neighbor 
                                found_next_step = True
                                break
                    
                    if path_complete:
                        break

                    if not found_next_step:
                        print(f"Warning: Could not complete path reconstruction for instance. Stuck at node {current_node}. Adjacency: {adj_solution[current_node]}. Current path: {reconstructed_path}")
                        return 0.0, [0]

                # Check if the reconstructed path is valid (starts and ends at depot, and has at least one other node)
                if reconstructed_path and reconstructed_path[0] == 0 and reconstructed_path[-1] == 0 and len(reconstructed_path) > 1:
                    # The `solve` method expects a list of 1-indexed nodes visited between depots.
                    # `reconstructed_path` is [0, n1, n2, ..., nk, 0]
                    # We want [n1, n2, ..., nk] for calc_op_length and calc_op_total.
                    nodes_in_tour_no_depot = reconstructed_path[1:-1]
                    
                    # Calculate the tour length using Gurobi's own 'dist' values
                    # gurobi_calculated_tour_length = gp.quicksum(x[i,j] * dist[i,j] for i,j in selected_edges_raw).getValue()
                    # print(f"Gurobi's calculated length for this solution: {gurobi_calculated_tour_length}")
                    # print(f"Max length allowed: {max_length}")
                    
                    # Calculate actual objective value from the reconstructed tour and prizes.
                    # The `model.objVal` is based on minimizing negative prizes.
                    actual_prize_sum = sum(prize[node_idx - 1] for node_idx in nodes_in_tour_no_depot)
                    
                    # Compare with Gurobi's objective.
                    if not np.isclose(model.objVal, -actual_prize_sum, atol=1e-4):
                        print(f"Warning: Gurobi objective {model.objVal} does not perfectly match calculated prize {-actual_prize_sum}.")
                        print("This might be due to numerical precision, or Gurobi finding an incumbent that is not the absolute best, but within tolerance.")
                    
                    return actual_prize_sum, nodes_in_tour_no_depot

                else:
                    print(f"Error: Reconstructed path is not a valid tour (starts/ends at depot, non-trivial). Path: {reconstructed_path}")
                    return 0.0, [0]

            except gp.GurobiError as e:
                print(f"Warning: Failed to retrieve solution for instance: {e}")
            
        return 0.0, [0]

    def solve(
        self, 
        depot: Union[list, np.ndarray] = None,
        loc: Union[list, np.ndarray] = None,
        prize: Union[list, np.ndarray] = None,
        max_length: Union[list, np.ndarray] = None,
        num_threads: int = 1, 
        show_time: bool = False, 
        gap: Optional[float] = None
    ) -> np.ndarray:
        """
        Solve OP instances using Gurobi
        
        :param depot: List of depot coordinates (n_instances, 2)
        :param loc: List of node coordinates (n_instances, n_nodes, 2)
        :param prize: List of node prizes (n_instances, n_nodes)
        :param max_length: List of maximum tour lengths (n_instances,)
        :param num_threads: Number of parallel threads to use
        :param show_time: Whether to show solving time
        :param gap: Optimality gap for the solver (default is 0.0, meaning no gap)
        :return: Tours for all instances
        """
        # Load data
        self.from_data(depots=depot, points=loc, prizes=prize, max_lengths=max_length)
        self.gap = gap
        
        n_instances = len(self.depots)
        tours = []
        total_costs = []
        
        # Solve each instance
        timer = Timer(apply=show_time)
        timer.start()

        if num_threads > 1:
            # Parallel processing
            with Pool(num_threads) as pool:
                results = []
                for i in range(n_instances):
                    args = (
                        self.depots[i],
                        self.points[i],
                        self.prizes[i],
                        self.max_lengths[i]
                    )
                    results.append(pool.apply_async(self._solve_single_instance, args))
                
                for i in range(n_instances):
                    # _solve_single_instance now returns (actual_prize_sum, [n1, n2, ..., nk])
                    cost, tour_nodes_1_indexed = results[i].get()
                    
                    # For `calc_op_length`, the tour input should be the 1-indexed nodes (as `tour_nodes_1_indexed` is).
                    calculated_length = self.calc_op_length(self.depots[i], self.points[i], tour_nodes_1_indexed)
                    assert calculated_length <= self.max_lengths[i] + MAX_LENGTH_TOL, \
                        f"Tour exceeds max_length! Instance {i}, Length: {calculated_length}, Max: {self.max_lengths[i]}"
                    
                    # For `calc_op_total`, the tour input should be the 1-indexed nodes (as `tour_nodes_1_indexed` is).
                    total_prize_calculated = self.calc_op_total(self.prizes[i], tour_nodes_1_indexed)
                    
                    assert abs(total_prize_calculated - cost) <= 1e-4, "Cost is incorrect"
                    
                    total_costs.append(total_prize_calculated)
                    tours.append(tour_nodes_1_indexed) # Store the 1-indexed nodes excluding depot
        else:
            # Sequential processing
            for i in range(n_instances):
                cost, tour_nodes_1_indexed = self._solve_single_instance(
                    self.depots[i],
                    self.points[i],
                    self.prizes[i],
                    self.max_lengths[i]
                )
                
                # Similar changes as above for sequential processing
                calculated_length = self.calc_op_length(self.depots[i], self.points[i], tour_nodes_1_indexed)
                print(f"Calculaxted length for instance {i}: {calculated_length}")
                assert calculated_length <= self.max_lengths[i] + MAX_LENGTH_TOL, \
                    f"Tour exceeds max_length! Instance {i}, Length: {calculated_length}, Max: {self.max_lengths[i]}"
                
                total_prize_calculated = self.calc_op_total(self.prizes[i], tour_nodes_1_indexed)
                assert abs(total_prize_calculated - cost) <= 1e-4, "Cost is incorrect"
                
                total_costs.append(total_prize_calculated)
                tours.append(tour_nodes_1_indexed) # Store the 1-indexed nodes excluding depot
        
        # Store results
        self.from_data(tours=tours, ref=False)
        
        timer.end()
        timer.show_time()
        
        return total_costs, tours

    def __str__(self) -> str:
        return "OPGurobiSolver"