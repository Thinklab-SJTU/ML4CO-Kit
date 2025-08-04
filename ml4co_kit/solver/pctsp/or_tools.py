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


import math
import numpy as np
from multiprocessing import Pool
from typing import Union, Tuple, List
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver.routing_enums_pb2 import LocalSearchMetaheuristic
from ml4co_kit.solver.pctsp.base import PCTSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class PCTSPORSolver(PCTSPSolver):
    r"""
    Solve PCTSPs using OR-Tools.

    :param scale, int, the scale factor for coordinates.
    :param time_limit: int, the limit of running time.
    :param strategy, str, choices: ["auto", "greedy", "guided", "guided", "tabu", "generic_tabu"]
    """
    def __init__(
        self, 
        scale: int = 1e6, 
        strategy: str = "guided", 
        time_limit: int = 1,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(PCTSPORSolver, self).__init__(
            solver_type=SOLVER_TYPE.ORTOOLS, scale=scale, precision=precision
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
        
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.search_parameters.local_search_metaheuristic = (meta_heu_dict[self.strategy])
        self.search_parameters.time_limit.seconds = self.time_limit
        self.search_parameters.log_search = False

    def _solve(self, 
        depots: np.ndarray,
        points: np.ndarray,
        penalties: np.ndarray,
        norm_prizes: np.ndarray,
    ) -> Tuple[float, List[int]]:
        r"""
        Solve a single PCTSP instance using OR-Tools.
        """
        # 1. Prepare data for OR-Tools, scaling coordinates, prizes, penalties
        coords = np.vstack([depots, points])
        coords = np.round(coords * self.scale).astype(np.int32)
        penalties = np.round(penalties * self.scale).astype(np.int32)
        norm_prizes = np.round(norm_prizes * self.scale).astype(np.int32)

        # 2. Create Routing Index Manager
        manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)

        # 3. Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # 4. Define Arc Cost (Distance)
        distance_evaluator_obj = _CreateDistanceEvaluator(coords, manager)
        transit_callback_index = routing.RegisterTransitCallback(distance_evaluator_obj.distance_evaluator)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 5. Add Penalties for Unvisited Nodes (Disjunctions)
        # Note: Nodes for disjunctions are 1-indexed (excluding depot) relative to their original problem indices
        # In OR-Tools, the node index `c + 1` corresponds to the actual location index.
        # Penalties are applied to nodes from 1 to num_locations - 1
        for i, p_val in enumerate(penalties):
            # routing.AddDisjunction takes a list of node indices (managed by manager)
            # The actual node index for the i-th penalty (0-indexed penalty array) is (i + 1)
            # manager.NodeToIndex converts the problem node index to the internal routing index
            routing.AddDisjunction([manager.NodeToIndex(i + 1)], int(p_val))
            
        # 6. Add Minimum Total Prize Constraint (Cumulative Dimension)
        prize_evaluator_obj = _CreatePrizeEvaluator(norm_prizes, manager)
        prize_callback_index = routing.RegisterTransitCallback(prize_evaluator_obj.prize_evaluator)
        
        prize_dimension_name = 'Prize'
        routing.AddDimension(
            prize_callback_index,
            0,  # null capacity slack (no prize accumulated when not moving)
            int(norm_prizes.sum()), # total capacity, i.e., max possible prize
            True,  # start cumul to zero
            prize_dimension_name
        )
        prize_dimension = routing.GetDimensionOrDie(prize_dimension_name)
        prize_dimension.CumulVar(routing.End(0)).RemoveInterval(0, int(self.scale - 1))

        # 7. Solve the problem
        solution = routing.SolveWithParameters(self.search_parameters)

        # 8. Extract solution
        if solution:
            # Reconstruct the route
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))

            # return route
            return np.array(route)
        else:
            raise ValueError(
                f"OR-Tools: No solution found for instance. Status: {routing.status()}"
            )
    
    def solve(
        self,
        depots: Union[np.ndarray, List[np.ndarray]] = None,
        points: Union[np.ndarray, List[np.ndarray]] = None,
        penalties: Union[np.ndarray, List[np.ndarray]] = None,
        norm_prizes: Union[np.ndarray, List[np.ndarray]] = None,
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
        :param norm_prizes: (num_instances, num_nodes), the prizes for each node for each instance (excluding depot).
        :param min_prizes: The minimum total prize required for each instance.
        :param norm: The normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        """
        # preparation
        self.from_data(
            depots=depots, points=points, penalties=penalties, 
            norm_prizes=norm_prizes, norm=norm, normalize=normalize
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
                    self.penalties[idx],
                    self.norm_prizes[idx]
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
                             self.penalties[idx*num_threads+inner_idx],
                             self.norm_prizes[idx*num_threads+inner_idx])
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
        return "PCTSPORSolver"
    

# Euclidean distance function (scaled)
def _euclidean_distance_scaled(position_1: np.ndarray, position_2: np.ndarray) -> int:
    r"""Computes the Euclidean distance between two scaled integer points."""
    p1 = position_1.astype(np.int64)
    p2 = position_2.astype(np.int64)
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return int(math.sqrt((dx ** 2 + dy ** 2) + 0.5))


class _CreateDistanceEvaluator(object):
    r"""Creates callback to return distance between points."""
    def __init__(self, locations_scaled: List[Tuple[int, int]], manager: pywrapcp.RoutingIndexManager):
        self._distances = {}
        self._manager = manager
        num_locations = len(locations_scaled)
        for from_node in range(num_locations):
            self._distances[from_node] = {}
            for to_node in range(num_locations):
                if from_node == to_node:
                    self._distances[from_node][to_node] = 0
                else:
                    self._distances[from_node][to_node] = (
                        _euclidean_distance_scaled(locations_scaled[from_node], locations_scaled[to_node]))

    def distance_evaluator(self, from_index: int, to_index: int) -> int:
        r"""Returns the scaled Euclidean distance between the two nodes."""
        # The 'from_index' and 'to_index' are solver indices, not node indices.
        from_node = self._manager.IndexToNode(from_index)
        to_node = self._manager.IndexToNode(to_index)
        return self._distances[from_node][to_node]


class _CreatePrizeEvaluator(object):
    r"""Creates callback to get prizes at each location."""
    def __init__(self, prizes_scaled: List[int], manager: pywrapcp.RoutingIndexManager):
        # We also need the manager here for index conversion.
        self._prizes = prizes_scaled
        self._manager = manager

    def prize_evaluator(self, from_index: int, to_index: int) -> int:
        r"""
        Returns the prize of the current node (0 for depot, prizes for others).
        """
        del to_index # to_node is not used for prize evaluation in PCTSP
        
        # The 'from_index' is a solver index. Convert it to a node index.
        from_node = self._manager.IndexToNode(from_index)
        
        # The rest of the logic is the same.
        # Depot (node index 0) has no prize.
        # Other nodes (node index i > 0) map to self._prizes[i - 1].
        return 0 if from_node == 0 else self._prizes[from_node - 1]