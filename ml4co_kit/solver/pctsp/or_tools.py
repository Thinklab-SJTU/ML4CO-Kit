r"""
OR-Tools for solving PCTSPs.

OR-Tools is open source software for combinatorial optimization, which seeks to 
find the best solution to a problem out of a very large set of possible solutions.

We follow https://developers.google.cn/optimization.
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
import math
from typing import Union, Tuple, List
from multiprocessing import Pool
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver.routing_enums_pb2 import LocalSearchMetaheuristic
from ml4co_kit.solver.pctsp.base import PCTSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.distance_utils import get_distance_matrix
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class PCTSPORSolver(PCTSPSolver):
    r"""
    Solve PCTSPs using OR-Tools.

    :param scale, int, the scale factor for coordinates.
    :param time_limit: int, the limit of running time.
    :param strategy, str, choices: ["auto", "greedy", "guided", "guided", "tabu", "generic_tabu"]
    """
    def __init__(
        self, scale: int = 1e6, strategy: str = "guided", time_limit: int = 1,
    ):
        super(PCTSPORSolver, self).__init__(
            solver_type=SOLVER_TYPE.ORTOOLS, scale=scale
        )
        self.strategy = strategy
        self.time_limit = time_limit
        self.set_search_parameters()
            
    def set_search_parameters(self):
        meta_heu_dict = {
            "auto": LocalSearchMetaheuristic.AUTOMATIC,
            "greedy": LocalSearchMetaheuristic.GREEDY_DESCENT,
            "guided": LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
            "simulated": LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            "tabu": LocalSearchMetaheuristic.TABU_SEARCH,
            "generic_tabu": LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
        }
        
        self.search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        self.search_parameters.local_search_metaheuristic = (meta_heu_dict[self.strategy])
        self.search_parameters.time_limit.seconds = self.time_limit
        self.search_parameters.log_search = False

    def _solve(self, 
        depot_coord: np.ndarray,
        nodes_coord: np.ndarray,
        penalties: np.ndarray,
        prizes: np.ndarray,
        min_prize: float = 1.0,
    ) -> Tuple[float, List[int]]:
        r"""
        Solve a single PCTSP instance using OR-Tools.

        :param depot_coord: (2,) array for depot coordinates.
        :param nodes_coord: (num_nodes, 2) array for other node coordinates.
        :param penalties: (num_nodes,) array for penalties for not visiting each node (excluding depot).
        :param prizes: (num_nodes,) array for prizes at each node (excluding depot).
        :param min_prize: float, the minimum total prize required.
        :return: Tuple[float, List[int]] -> (total_cost, route_nodes)
                 total_cost is the sum of tour length and unvisited penalties.
                 route_nodes is a list of 0-indexed nodes in the route, starting and ending at depot.
        """
        # 1. Prepare data for OR-Tools, scaling coordinates, prizes, penalties
        all_locations = [depot_coord] + nodes_coord.tolist() # Combine depot and nodes
        locations_scaled = [(_float_to_scaled_int(l[0]), _float_to_scaled_int(l[1])) for l in all_locations]

        prizes_scaled = [_float_to_scaled_int(v) for v in prizes]
        penalties_scaled = [_float_to_scaled_int(v) for v in penalties]
        
        # Ensure min_prize is feasible after scaling and rounding, as per the original example
        min_prize_scaled = _float_to_scaled_int(min_prize)
        total_possible_prize_scaled = sum(prizes_scaled)
        min_prize_scaled = min(min_prize_scaled, total_possible_prize_scaled) # Clamp if initial min_prize is too high


        num_locations = len(locations_scaled) # Total nodes including depot
        num_vehicles = 1
        depot_index = 0 # Depot is always node 0

        # 2. Create Routing Index Manager
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)

        # 3. Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # 4. Define Arc Cost (Distance)
        distance_evaluator_obj = _CreateDistanceEvaluator(locations_scaled)
        transit_callback_index = routing.RegisterTransitCallback(distance_evaluator_obj.distance_evaluator)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 5. Add Penalties for Unvisited Nodes (Disjunctions)
        # Note: Nodes for disjunctions are 1-indexed (excluding depot) relative to their original problem indices
        # In OR-Tools, the node index `c + 1` corresponds to the actual location index.
        # Penalties are applied to nodes from 1 to num_locations - 1
        for i, p_val in enumerate(penalties_scaled):
            # routing.AddDisjunction takes a list of node indices (managed by manager)
            # The actual node index for the i-th penalty (0-indexed penalty array) is (i + 1)
            # manager.NodeToIndex converts the problem node index to the internal routing index
            routing.AddDisjunction([manager.NodeToIndex(i + 1)], p_val)


        # 6. Add Minimum Total Prize Constraint (Cumulative Dimension)
        prize_evaluator_obj = _CreatePrizeEvaluator(prizes_scaled)
        prize_callback_index = routing.RegisterTransitCallback(prize_evaluator_obj.prize_evaluator)
        
        prize_dimension_name = 'Prize'
        routing.AddDimension(
            prize_callback_index,
            0,  # null capacity slack (no prize accumulated when not moving)
            total_possible_prize_scaled, # total capacity, i.e., max possible prize
            True,  # start cumul to zero
            prize_dimension_name
        )
        prize_dimension = routing.GetDimensionOrDie(prize_dimension_name)
        
        # This constrains the cumulative prize at the end of the route to be >= min_prize_scaled
        for vehicle_id in range(num_vehicles):
            # Remove interval [0, min_prize_scaled - 1]
            prize_dimension.CumulVar(routing.End(vehicle_id)).RemoveInterval(0, min_prize_scaled - 1)


        # 7. Solve the problem
        solution = routing.SolveWithParameters(self.search_parameters)

        # 8. Extract solution
        if solution:
            # Objective includes tour cost + total penalties for unvisited nodes
            total_cost_scaled = solution.ObjectiveValue()
            total_cost = total_cost_scaled / 10000000. 

            # Reconstruct the route
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))

            return total_cost, route # route will be [0, n1, n2, ..., nk, 0]
        else:
            # If no solution found, return default (e.g., very high cost, only depot)
            print(f"OR-Tools: No solution found for instance. Status: {routing.status()}")
            return float('inf'), [0, 0]
    
    def solve(
        self,
        depots: Union[np.ndarray, List[np.ndarray]],
        points: Union[np.ndarray, List[np.ndarray]],
        penalties: Union[np.ndarray, List[np.ndarray]],
        prizes: Union[np.ndarray, List[np.ndarray]],
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (costs, routes)
        r"""
        Solve PCTSP instances using OR-Tools.
        
        ::param depots: (num_instances, 2), the coordinates of depots for each instance.
        :param points: (num_instances, num_nodes, 2), the coordinates of nodes for each instance (excluding depot).
        :param penalties: (num_instances, num_nodes), the penalties for not visiting each node for each instance (excluding depot).
        :param prizes: (num_instances, num_nodes), the prizes for each node for each instance (excluding depot).
        :param min_prizes: The minimum total prize required for each instance.
        :param norm: The normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        """
        # preparation
        self.from_data(depots=depots, points=points, penalties=penalties, deterministic_prizes=prizes, norm=norm, normalize=normalize)
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        costs = list()
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                cost, tour = self._solve(
                    self.depots[idx],
                    self.points[idx],
                    self.penalties[idx],
                    self.deterministic_prizes[idx]
                )
                costs.append(cost)
                tours.append(tour)
        else:
            with Pool(num_threads) as p:
                results = p.starmap(self._solve, [(
                        self.depots[idx],
                        self.points[idx],
                        self.penalties[idx],
                        self.deterministic_prizes[idx]
                    ) for idx in range(num_points)
                ])
            for cost, tour in results:
                costs.append(cost)
                tours.append(tour)
        
        # format
        costs = np.array(costs)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return costs, self.tours

    def __str__(self) -> str:
        return "PCTSPORSolver"
    
    
def _float_to_scaled_int(v: float) -> int:
    r"""Scales a float to a large integer for OR-Tools."""
    return int(v * 10000000 + 0.5)


# Euclidean distance function (scaled)
def _euclidean_distance_scaled(position_1: Tuple[int, int], position_2: Tuple[int, int]) -> int:
    r"""Computes the Euclidean distance between two scaled integer points."""
    return int(math.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2) + 0.5)


class _CreateDistanceEvaluator(object):
    r"""Creates callback to return distance between points."""
    def __init__(self, locations_scaled: List[Tuple[int, int]]):
        self._distances = {}
        num_locations = len(locations_scaled)
        for from_node in range(num_locations):
            self._distances[from_node] = {}
            for to_node in range(num_locations):
                if from_node == to_node:
                    self._distances[from_node][to_node] = 0
                else:
                    self._distances[from_node][to_node] = (
                        _euclidean_distance_scaled(locations_scaled[from_node], locations_scaled[to_node]))

    def distance_evaluator(self, from_node, to_node):
        r"""Returns the scaled Euclidean distance between the two nodes."""
        return self._distances[from_node][to_node]
    
    
class _CreatePrizeEvaluator(object):
    r"""Creates callback to get prizes at each location."""
    def __init__(self, prizes_scaled: List[int]):
        self._prizes = prizes_scaled

    def prize_evaluator(self, from_node, to_node):
        r"""
        Returns the prize of the current node (0 for depot, prizes for others).
        Note: from_node is the routing index.
        """
        del to_node # to_node is not used for prize evaluation in PCTSP
        # Depot (routing index 0) has no prize
        # Other nodes (routing index i > 0) map to self._prizes[i - 1]
        return 0 if from_node == 0 else self._prizes[from_node - 1] # Prize is only for locations, not depot