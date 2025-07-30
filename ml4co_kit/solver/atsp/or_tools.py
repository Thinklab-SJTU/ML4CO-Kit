r"""
OR-Tools for solving ATSPs.

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
from ml4co_kit.solver.atsp.base import ATSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class ATSPORSolver(ATSPSolver):
    r"""
    Solve ATSPs using LKH solver.
    
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
        super(ATSPORSolver, self).__init__(
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
    
    def _solve(self, dist: np.ndarray) -> np.ndarray:
        # construct the TSP data dict to solve
        data = {}
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
        tour.append(0)
        return tour
    
    def solve(
        self,
        dists: Union[np.ndarray, list] = None,
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        r"""
        Solve ATSP using LKH algorithm with options for normalization,
        threading, and timing.

        :param dists: np.ndarry or list, the dists matrix to solve TSP for. Defaults to None.
        :param normalize: boolean, whether to normalize the points. Defaults to "False".
        :param num_threads: int, The number of threads to use for solving. Defaults to 1.
        :param show_time: boolean, whether to show the time taken to solve. Defaults to "False".

        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import ATSPLKHSolver
                
                # create ATSPLKHSolver
                >>> solver = ATSPLKHSolver(lkh_max_trials=1)

                # load data and reference solutions from ``.tsp`` file
                >>> solver.from_tsplib(
                        atsp_file_path="examples/atsp/tsplib_1/problem/ft53.atsp",
                        ref=False,
                        normalize=True
                    )
                    
                # solve
                >>> solver.solve()
                [[ 0,  5,  9, 11,  3,  8,  4,  1,  2, 10,  7,  6,  0]]
         """
    
        # prepare
        self.from_data(dists=dists, normalize=normalize)
        self.tmp_solver = ATSPSolver(scale=self.scale)
        timer = Timer(apply=show_time)
        timer.start()

        # solve
        tours = list()
        dists_shape = self.dists.shape
        num_dists = dists_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_dists, self.solve_msg, show_time):
                tours.append(self._solve(self.dists[idx]))
        else:
            batch_dists = self.dists.reshape(-1, num_threads, dists_shape[-2], dists_shape[-1])
            for idx in iterative_execution(
                range, num_dists // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.map(
                        self._solve,
                        [
                            batch_dists[idx][inner_idx]
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
        
        return self.tours
    
    def __str__(self) -> str:
        return "ATSPORSolver"