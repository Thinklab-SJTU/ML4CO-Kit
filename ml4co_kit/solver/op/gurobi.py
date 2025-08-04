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

import itertools
import numpy as np
import gurobipy as gp
from multiprocessing import Pool
from typing import Union, Tuple, List
from ml4co_kit.solver.op.base import OPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class OPGurobiSolver(OPSolver):
    def __init__(
        self, 
        scale: int = 1e6, 
        time_limit: float = 60.0, 
        gurobi_gap: float = 0.0,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(OPGurobiSolver, self).__init__(
            solver_type=SOLVER_TYPE.GUROBI, scale=scale, precision=precision
        )
        self.time_limit = time_limit
        self.gurobi_gap = gurobi_gap
        
    def _solve(
        self, 
        depots: np.ndarray, 
        points: np.ndarray, 
        prizes: np.ndarray, 
        max_length: float
    ) -> Tuple[float, List[int]]:
        """
        Solve a single OP instance using Gurobi
        """
        # preparation
        coords = np.vstack((depots[None, :], points))
        nodes_num = len(coords)

        # callback - use lazy constraints to eliminate sub-tours
        def subtourelim(model, where):
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
        dist = {(i,j) : np.linalg.norm(coords[i] - coords[j])
                for i, j in itertools.combinations(range(nodes_num), 2)}

        # Create Gurobi model
        model = gp.Model()
        model.Params.outputFlag = False

        # Create variables
        x = model.addVars(itertools.combinations(range(nodes_num), 2), vtype=gp.GRB.BINARY, name='x')
        
        # Node selection variables (delta_i = 1 if node i is visited)
        prize_dict = {
            i + 1: -p  # We need to maximize so negate
            for i, p in enumerate(prizes)
        }
        
        # delta variables are for nodes 1 to n-1 (since node 0 is depot and always visited)
        delta = model.addVars(range(1, nodes_num), obj=prize_dict, vtype=gp.GRB.BINARY, name='delta')
        
        # Degree constraints
        # For depot (node 0): sum of x_0j = 2   
        model.addConstr(gp.quicksum(x[0, j] for j in range(1, nodes_num)) == 2, name='depot_degree')

        # For other nodes i (1 to n-1): sum of x_ij = 2 * delta_i
        for i in range(1, nodes_num):
            # Sum of edges connected to node i where i < j: x[i,j]
            # Sum of edges connected to node i where k < i: x[k,i]
            model.addConstr(gp.quicksum(x[min(i,j), max(i,j)] for j in range(nodes_num) if i != j) == 2 * delta[i], name=f'degree_{i}')

        # Tour length constraint
        model.addConstr(gp.quicksum(x[i, j] * dist[i, j] for i, j in itertools.combinations(range(nodes_num), 2)) <= max_length, name='max_length')

        # Set model references for callback
        model._vars = x # Now _vars are the x_ij variables
        model._dvars = delta
        
        # Set parameters
        model.Params.lazyConstraints = 1
        model.Params.threads = 1 # Keep 1 thread for callback
        if self.time_limit:
            model.Params.timeLimit = self.time_limit
        if self.gurobi_gap:
            model.Params.mipGap = self.gurobi_gap * 0.01  # Percentage

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
                
                # Reconstruct the tour path (starts and ends at depot 0)
                # Build an adjacency list from the selected edges
                adj_solution = [[] for _ in range(nodes_num)] # n is total nodes including depot
                for i, j in selected_edges_raw:
                    adj_solution[i].append(j)
                    adj_solution[j].append(i) # Add both directions for traversal

                reconstructed_path = []
                
                # Verify depot's degree
                if len(adj_solution[0]) != 2:
                    # This indicates a problem with the Gurobi solution if it's supposed to be a tour.
                    # It implies the degree constraint on depot (sum(x_0j) == 2) was violated or not fully enforced.
                    warn_msg = (
                        "Warning: Depot (node 0) does not have degree 2 in the extracted solution. "
                        f"Actual degree: {len(adj_solution[0])}"
                    )
                    print(warn_msg)
                    # Attempt to find *any* path if degree is not 2, or return default
                    return [0] 

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
                    return [0]

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
                        warn_msg = (
                            f"Warning: Could not complete path reconstruction for instance. "
                            f"Stuck at node {current_node}. Adjacency: {adj_solution[current_node]}. "
                            f"Current path: {reconstructed_path}"
                        )
                        print(warn_msg)
                        return [0]

                # Check if the reconstructed path is valid (starts and ends at depot, and has at least one other node)
                if reconstructed_path and reconstructed_path[0] == 0 and reconstructed_path[-1] == 0 and len(reconstructed_path) > 1:
                    # The `solve` method expects a list of 1-indexed nodes visited between depots.
                    # `reconstructed_path` is [0, n1, n2, ..., nk, 0]
                    # We want [n1, n2, ..., nk] for calc_op_length and calc_op_total.
                    nodes_in_tour_no_depot = reconstructed_path[1:-1]
                    
                    # Calculate actual objective value from the reconstructed tour and prizes.
                    # The `model.objVal` is based on minimizing negative prizes.
                    actual_prize_sum = sum(prizes[node_idx - 1] for node_idx in nodes_in_tour_no_depot)
                    
                    # Compare with Gurobi's objective.
                    if not np.isclose(model.objVal, -actual_prize_sum, atol=1e-4):
                        warn_msg = (
                            f"Warning: Gurobi objective {model.objVal} does not perfectly "
                            f"match calculated prize {-actual_prize_sum}. This might be due to " 
                            "numerical precision, or Gurobi finding an incumbent that is not "
                            "the absolute best, but within tolerance."
                        )
                        print(warn_msg)

                    return nodes_in_tour_no_depot

                else:
                    error_msg = (
                        "Error: Reconstructed path is not a valid tour"
                        f"(starts/ends at depot, non-trivial). Path: {reconstructed_path}" 
                    )
                    print(error_msg)
                    return [0]

            except gp.GurobiError as e:
                print(f"Warning: Failed to retrieve solution for instance: {e}")
            
        return [0]

    def solve(
        self, 
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        prizes: Union[list, np.ndarray] = None,
        max_lengths: Union[list, np.ndarray] = None,
        num_threads: int = 1, 
        show_time: bool = False, 
    ) -> np.ndarray:
        """
        Solve OP instances using Gurobi
        """
        # preparation
        self.from_data(
            depots=depots, points=points, prizes=prizes, max_lengths=max_lengths
        )
        timer = Timer(apply=show_time)
        timer.start()

        # solve
        tours = list()
        num_points = self.points.shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                tour = self._solve(
                    self.depots[idx],
                    self.points[idx],
                    self.prizes[idx],
                    self.max_lengths[idx]
                )
                tours.append(tour)
        else:
            for idx in iterative_execution(
                range, num_points // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.starmap(
                        self._solve,
                        [
                            (self.depots[idx*num_threads+inner_idx],
                             self.points[idx*num_threads+inner_idx],
                             self.prizes[idx*num_threads+inner_idx],
                             self.max_lengths[idx*num_threads+inner_idx])
                            for inner_idx in range(num_threads)
                        ],
                    )
                for tour in cur_tours:
                    tours.append(tour)

        # format
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return self.tours

    def __str__(self) -> str:
        return "OPGurobiSolver"