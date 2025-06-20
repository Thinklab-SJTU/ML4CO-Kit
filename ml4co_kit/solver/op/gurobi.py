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
        points = [depot] + loc
        n = len(points)

        # Callback - use lazy constraints to eliminate sub-tours
        def subtourelim(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                # Make a list of edges selected in the solution
                vals = model.cbGetSolution(model._vars)
                selected = gp.tuplelist(
                    (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
                )
                # Find the shortest cycle in the selected edge list
                tour = subtour(selected)
                if tour is not None:
                    # Add subtour elimination constraint
                    model.cbLazy(
                        gp.quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2))
                        <= gp.quicksum(model._dvars[i] for i in tour) * (len(tour) - 1) / float(len(tour))
                    )

        # Given a tuplelist of edges, find the shortest subtour
        def subtour(edges, exclude_depot=True):
            unvisited = list(range(n))
            cycle = None
            while unvisited:
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
                # Keep this cycle if it's shorter and doesn't exclude depot
                if (
                    (cycle is None or len(cycle) > len(thiscycle))
                        and len(thiscycle) > 1 and not (0 in thiscycle and exclude_depot)
                ):
                    cycle = thiscycle
            return cycle

        # Calculate Euclidean distances between each pair of points
        dist = {(i,j) :
            math.sqrt(sum((points[i][k] - points[j][k])**2 for k in range(2)))
            for i in range(n) for j in range(i)}

        # Create Gurobi model
        model = gp.Model()
        model.Params.outputFlag = False

        # Create variables
        vars = model.addVars(dist.keys(), vtype=gp.GRB.BINARY, name='e')
        for i,j in list(vars.keys()):
            vars[j,i] = vars[i,j] # edge in opposite direction
        
        # Depot variables can have value 2 (since it's start and end)
        for i, j in vars.keys():
            if i == 0 or j == 0:
                vars[i, j].vtype = gp.GRB.INTEGER
                vars[i, j].ub = 2

        # Node selection variables (delta_i = 1 if node i is visited)
        prize_dict = {
            i + 1: -p  # We need to maximize so negate
            for i, p in enumerate(prize)
        }
        delta = model.addVars(range(1, n), obj=prize_dict, vtype=gp.GRB.BINARY, name='delta')

        # Degree constraints
        model.addConstrs(vars.sum(i,'*') == (2 if i == 0 else 2 * delta[i]) for i in range(n))

        # Tour length constraint
        model.addConstr(gp.quicksum(var * dist[i, j] for (i, j), var in vars.items() if j < i) <= max_length)

        # Set model references for callback
        model._vars = vars
        model._dvars = delta
        
        # Set parameters
        model.Params.lazyConstraints = 1
        model.Params.threads = 1
        if self.time_limit:
            model.Params.timeLimit = self.time_limit
        if self.gap:
            model.Params.mipGap = self.gap * 0.01  # Percentage

        # Optimize model
        model.optimize(subtourelim)

        # Extract solution
        print(model.status)
        if model.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SOLUTION_LIMIT]:
            try:
                vals = model.getAttr('x', vars)
                selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
                tour = subtour(selected, exclude_depot=False)
                if tour is not None and tour[0] == 0:
                    return model.objVal, tour
            except gp.GurobiError:
                print(f"Warning: Failed to retrieve solution for instance")
            
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
                    cost, tour = results[i].get()
                    assert tour[0] == 0, "Tour must start from depot"
                    tour = tour[1:]  # Remove depot from tour
                    assert self.calc_op_length(self.depots[i], self.points[i], tour) <= self.max_lengths[i] + MAX_LENGTH_TOL, "Tour exceeds max_length!"
                    total_cost = -self.calc_op_total(self.prizes[i], tour)
                    assert abs(total_cost - cost) <= 1e-4, "Cost is incorrect"
                    total_costs.append(total_cost)
                    tours.append(tour)
        else:
            # Sequential processing
            for i in range(n_instances):
                cost, tour = self._solve_single_instance(
                    self.depots[i],
                    self.points[i],
                    self.prizes[i],
                    self.max_lengths[i]
                )
                assert tour[0] == 0, "Tour must start from depot"
                tour = tour[1:]  # Remove depot from tour
                print(self.calc_op_length(self.depots[i], self.points[i], tour))
                assert self.calc_op_length(self.depots[i], self.points[i], tour) <= self.max_lengths[i] + MAX_LENGTH_TOL, "Tour exceeds max_length!"
                total_cost = -self.calc_op_total(self.prizes[i], tour)
                assert abs(total_cost - cost) <= 1e-4, "Cost is incorrect"
                total_costs.append(total_cost)
                tours.append(tour)
        
        # Store results
        self.from_data(tours=tours, ref=False)
        
        timer.end()
        timer.show_time()
        
        return total_costs, tours

    def __str__(self) -> str:
        return "OPGurobiSolver"