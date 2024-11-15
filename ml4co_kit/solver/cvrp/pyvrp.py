import sys
import time
import numpy as np
from typing import Union
from multiprocessing import Pool
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from ml4co_kit.solver.cvrp.base import CVRPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


if sys.version_info.major == 3 and sys.version_info.minor == 8:
    CP38 = True
else:
    CP38 = False


class CVRPPyVRPSolver(CVRPSolver):
    def __init__(
        self,
        depots_scale: int = 1e4,
        points_scale: int = 1e4,
        demands_scale: int = 1e3,
        capacities_scale: int = 1e3,
        time_limit: float = 1.0,
    ):
        super(CVRPPyVRPSolver, self).__init__(
            solver_type=SOLVER_TYPE.PYVRP, 
            depots_scale = depots_scale,
            points_scale = points_scale,
            demands_scale = demands_scale,
            capacities_scale = capacities_scale,
        )
        self.time_limit = time_limit

    def _solve(
        self, 
        depot_coord: np.ndarray, 
        nodes_coord: np.ndarray,
        demands: np.ndarray,
        capacity: float
    ) -> list:
        # scale
        depot_coord = (depot_coord * self.depots_scale).astype(np.int64)
        nodes_coord = (nodes_coord * self.points_scale).astype(np.int64)
        demands = (demands * self.demands_scale).astype(np.int64)
        capacity = int(capacity * self.capacities_scale)
        
        # solve
        cvrp_model = Model()
        depot = cvrp_model.add_depot(x=depot_coord[0], y=depot_coord[1])
        max_num_available = len(demands)
        cvrp_model.add_vehicle_type(capacity=capacity, num_available=max_num_available)
        clients = [
            cvrp_model.add_client(
                int(self.round_func(nodes_coord[idx][0])), 
                int(self.round_func(nodes_coord[idx][1])), 
                int(self.round_func(demands[idx]))
            ) for idx in range(0, len(nodes_coord))
        ]
        locations = [depot] + clients
        for frm in locations:
            for to in locations:
                distance = self.get_distance(x1=(frm.x, frm.y), x2=(to.x, to.y))
                cvrp_model.add_edge(frm, to, distance=self.round_func(distance))
        res = cvrp_model.solve(stop=MaxRuntime(self.time_limit))
        
        routes = res.best.get_routes() if CP38 else res.best.routes()
        tour = [0]
        for route in routes:
            tour += route.visits()
            tour.append(0)
        return tour
        
    def solve(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
        round_func: str = "round",
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        # preparation
        self.from_data(
            depots=depots, points=points, demands=demands,
            capacities=capacities, norm=norm, normalize=normalize
        )
        self.round_func = self.get_round_func(round_func)
        timer = Timer(apply=show_time)
        timer.start()

        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:   
            for idx in iterative_execution(
                range, num_points, "Solving CVRP Using PyVRP", show_time
            ):
                tours.append(self._solve(
                    depot_coord=self.depots[idx],
                    nodes_coord=self.points[idx],
                    demands=self.demands[idx],
                    capacity=self.capacities[idx]
                ))
        else:
            num_tqdm = num_points // num_threads
            batch_depots = self.depots.reshape(num_tqdm, num_threads, -1)
            batch_demands = self.demands.reshape(num_tqdm, num_threads, -1)
            batch_capacities = self.capacities.reshape(num_tqdm, num_threads)
            batch_points = self.points.reshape(-1, num_threads, p_shape[-2], p_shape[-1])
            for idx in iterative_execution(
                range, num_points // num_threads, "Solving CVRP Using PyVRP", show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.starmap(
                        self._solve,
                        [  (batch_depots[idx][inner_idx], 
                            batch_points[idx][inner_idx], 
                            batch_demands[idx][inner_idx], 
                            batch_capacities[idx][inner_idx]) 
                            for inner_idx in range(num_threads)
                        ],
                    )
                for tour in cur_tours:
                    tours.append(tour)

        # format
        self.from_data(tours=tours)
        
        # show time
        timer.end()
        timer.show_time()

        return self.tours
    
    def __str__(self) -> str:
        return "CVRPPyVRPSolver"