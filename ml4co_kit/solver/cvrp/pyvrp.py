import sys
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from typing import Union
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from .base import CVRPSolver


if sys.version_info.major == 3 and sys.version_info.minor == 8:
    CP38 = True
else:
    CP38 = False


class CVRPPyVRPSolver(CVRPSolver):
    def __init__(
        self,
        depots_scale: int = 1e6,
        points_scale: int = 1e6,
        demands_scale: int = 1e6,
        capacities_scale: int = 1e6,
        time_limit: float = 1.0,
    ):
        super(CVRPPyVRPSolver, self).__init__(
            solver_type="pyvrp", 
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
                self.round_func(nodes_coord[idx][0]), 
                self.round_func(nodes_coord[idx][1]), 
                self.round_func(demands[idx])
            ) for idx in range(0, len(nodes_coord))
        ]
        locations = [depot] + clients
        for frm in locations:
            for to in locations:
                distance = self.get_distance(x1=(frm.x, frm.y), x2=(to.x, to.y))
                cvrp_model.add_edge(frm, to, distance=self.round_func(distance))
        res = cvrp_model.solve(stop=MaxRuntime(self.time_limit))
        
        routes = res.best.get_routes() if CP38 else res.best.routes()
        tours = [0]
        for route in routes:
            tours += route.visits()
            tours.append(0)
        return tours
        
    def solve(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        dtype: str = "int",
        round_func: str = "round",
        num_threads: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        # prepare
        if dtype != "int":
            import warnings
            warnings.warn("Solver input requires data of type int.")
            dtype = "int"
        self.round_func = self.get_round_func(round_func)
        self.from_data(depots, points, demands, capacities, norm, normalize)
        start_time = time.time()

        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            if show_time:
                for idx in tqdm(range(num_points), desc="Solving CVRP Using PyVRP"):
                    tours.append(self._solve(
                        depot_coord=self.depots[idx],
                        nodes_coord=self.points[idx],
                        demands=self.demands[idx],
                        capacity=self.capacities[idx]
                    ))
            else:
                for idx in range(num_points):
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
            if show_time:
                for idx in tqdm(
                    range(num_points // num_threads), desc="Solving CVRP Using PyVRP"
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
            else:
                for idx in range(num_points // num_threads):
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
        self.read_tours(tours)
        end_time = time.time()
        if show_time:
            print(f"Use Time: {end_time - start_time}")
        return self.tours