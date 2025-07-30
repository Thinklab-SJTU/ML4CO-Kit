r"""
OR-Tools for solving TSPs.

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
from typing import Union
from multiprocessing import Pool
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver.routing_enums_pb2 import LocalSearchMetaheuristic
from ml4co_kit.solver.tsp.base import TSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.distance_utils import get_distance_matrix
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class TSPORSolver(TSPSolver):
    r"""
    Solve TSPs using OR-Tools.

    :param scale, int, the scale factor for coordinates.
    :param time_limit: int, the limit of running time.
    :param strategy, str, choices: ["auto", "greedy", "guided", "guided", "tabu", "generic_tabu"]
    """
    def __init__(
        self, scale: int = 1e6, 
        strategy: str = "guided", 
        time_limit: int = 1,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(TSPORSolver, self).__init__(
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

    def _solve(self, nodes_coord: np.ndarray) -> list:
        r"""
        Solve a single TSP instance
        """
        # construct the TSP data dict to solve
        data = {}
        dist = get_distance_matrix(points=nodes_coord, norm=self.norm)
        dist = (dist * self.scale).astype(np.int32)
        data["distance_matrix"] = dist
        data["num_vehicles"] = 1
        data["depot"] = 0

        # create manager
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Solve the problem.
        solution = routing.SolveWithParameters(self.search_parameters)
        
        # solution -> tour
        tour = list()
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            index = solution.Value(routing.NextVar(index))
            tour.append(node)
        return tour
    
    def solve(
        self,
        points: Union[np.ndarray, list] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        r"""
        :param points: np.ndarray, the coordinates of nodes. If given, the points 
            originally stored in the solver will be replaced.
        :param norm: boolean, the normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        """
        # preparation
        self.from_data(points=points, norm=norm, normalize=normalize)
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                tours.append(self._solve(self.points[idx]))
        else:
            batch_points = self.points.reshape(-1, num_threads, p_shape[-2], p_shape[-1])
            for idx in iterative_execution(
                range, num_points // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.map(
                        self._solve,
                        [
                            batch_points[idx][inner_idx]
                            for inner_idx in range(num_threads)
                        ],
                    )
                for tour in cur_tours:
                    tours.append(tour)
        
        # format
        tours = np.array(tours)
        zeros = np.zeros((tours.shape[0], 1))
        tours = np.append(tours, zeros, axis=1).astype(np.int32)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return self.tours

    def __str__(self) -> str:
        return "TSPORSolver"