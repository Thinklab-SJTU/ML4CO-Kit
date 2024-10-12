import os
import uuid
import time
import numpy as np
from typing import Union
from multiprocessing import Pool
from ml4co_kit.solver.cvrp.base import CVRPSolver
from ml4co_kit.solver.cvrp.c_hgs import cvrp_hgs_solver, HGS_TMP_PATH
from ml4co_kit.utils.run_utils import iterative_execution


class CVRPHGSSolver(CVRPSolver):
    def __init__(
        self,
        depots_scale: int = 2e4,
        points_scale: int = 2e4,
        demands_scale: int = 1e3,
        capacities_scale: int = 1e3,
        time_limit: float = 1.0,
        show_info: bool = False
    ):
        super(CVRPHGSSolver, self).__init__(
            solver_type="HGS", 
            depots_scale = depots_scale,
            points_scale = points_scale,
            demands_scale = demands_scale,
            capacities_scale = capacities_scale,
        )
        self.time_limit = time_limit
        self.show_info = show_info

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
        
        # generate .vrp file
        name = uuid.uuid4().hex[:9]
        tmp_solver = CVRPSolver()
        tmp_solver.from_data(depot_coord, nodes_coord, demands, capacity)
        tmp_solver.to_vrplib_folder(vrp_save_dir=HGS_TMP_PATH, vrp_filename=name)
        
        # Intermediate files
        vrp_name = f"{name}.vrp"
        sol_name = f"{name}.sol"
        vrp_abs_path = os.path.join(HGS_TMP_PATH, vrp_name)
        sol_abs_path = os.path.join(HGS_TMP_PATH, sol_name)
        pg_abs_path = os.path.join(HGS_TMP_PATH, f"{name}.sol.PG.csv")
        
        # solve
        cvrp_hgs_solver(vrp_name, sol_name, self.time_limit, self.show_info)
        
        # read data from .sol
        tmp_solver.from_vrplib(sol_file_path=sol_abs_path, ref=False)
        tour = tmp_solver.tours[0]
        
        # clear files
        intermediate_files = [vrp_abs_path, sol_abs_path, pg_abs_path]
        for file_path in intermediate_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return tour
        
    def solve(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
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
        
        # start time
        start_time = time.time()

        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:   
            for idx in iterative_execution(
                range, num_points, "Solving CVRP Using HGS", show_time
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
                range, num_points // num_threads, "Solving CVRP Using HGS", show_time
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
        self.from_data(tours=tours, ref=False)
        end_time = time.time()
        if show_time:
            print(f"Use Time: {end_time - start_time}")
        return self.tours
    
    def __str__(self) -> str:
        return "CVRPHGSSolver"