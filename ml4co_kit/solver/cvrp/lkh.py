import os
import uuid
import pathlib
import numpy as np
from typing import Union
from multiprocessing import Pool
from subprocess import check_call
from ml4co_kit.utils import tsplib95
from ml4co_kit.solver.cvrp.base import CVRPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class CVRPLKHSolver(CVRPSolver):
    def __init__(
        self,
        depots_scale: int = 1e4,
        points_scale: int = 1e4,
        demands_scale: int = 1e3,
        capacities_scale: int = 1e3,
        lkh_max_trials: int = 500,
        lkh_path: pathlib.Path = "LKH",
        lkh_runs: int = 1,
        lkh_seed: int = 1234,
        lkh_special: bool = True
    ):
        """
        TSPLKHSolver
        Args:
            lkh_max_trials (int, optional): The maximum number of trials for
                the LKH solver. Defaults to 500.
            lkh_path (pathlib.Path, optional): The path to the LKH solver.
                Defaults to "LKH".
            scale (int, optional): The scale factor for coordinates in the
                LKH solver. Defaults to 1e6.
            lkh_runs (int, optional): The number of runs for the LKH solver.
                Defaults to 1.
        """
        super(CVRPLKHSolver, self).__init__(
            solver_type=SOLVER_TYPE.LKH, 
            depots_scale = depots_scale,
            points_scale = points_scale,
            demands_scale = demands_scale,
            capacities_scale = capacities_scale,
        )
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_runs = lkh_runs
        self.lkh_seed = lkh_seed
        self.lkh_special = lkh_special

    def write_parameter_file(
        self,
        save_path: str,
        vrp_file_path: str,
        tour_path: str
    ):
        with open(save_path, "w") as f:
            f.write(f"PROBLEM_FILE = {vrp_file_path}\n")
            f.write(f"MAX_TRIALS = {self.lkh_max_trials}\n")
            if self.lkh_special:
                f.write("SPECIAL\n")
            f.write(f"RUNS = {self.lkh_runs}\n")
            f.write(f"SEED = {self.lkh_seed}\n")
            f.write(f"TOUR_FILE = {tour_path}\n")
    
    def read_lkh_solution(self, tour_path: str) -> list:
        tour = tsplib95.load(tour_path).tours[0]
        np_tour = np.array(tour) - 1
        over_index = np.where(np_tour > self.nodes_num)[0]
        np_tour[over_index] = 0
        tour = np_tour.tolist()
        tour: list
        tour.append(0)
        return tour
        
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
        
        # Intermediate files
        tmp_name = uuid.uuid4().hex[:9]
        para_save_path = f"{tmp_name}.para"
        vrp_save_path = f"{tmp_name}.vrp"
        tour_save_path = f"{tmp_name}.tour"
        log_save_path = f"{tmp_name}.log"
        
        # prepare for solve
        self.tmp_solver.from_data(
            depots=depot_coord, points=nodes_coord, 
            demands=demands, capacities=capacity
        )
        self.tmp_solver.to_vrplib_folder(
            vrp_save_dir="./", vrp_filename=vrp_save_path
        )
        self.write_parameter_file(
            save_path=para_save_path,
            vrp_file_path=vrp_save_path,
            tour_path=tour_save_path
        )
        
        # solve
        with open(log_save_path, "w") as f:
            check_call([self.lkh_path, para_save_path], stdout=f)
            
        # read solution
        tour = self.read_lkh_solution(tour_save_path)
        
        # delete files
        files_path = [
            para_save_path, vrp_save_path,
            tour_save_path, log_save_path
        ]
        for file_path in files_path:
           if os.path.exists(file_path):
               os.remove(file_path)
        
        # return
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
        self.tmp_solver = CVRPSolver()
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(
                range, num_points, "Solving CVRP Using LKH", show_time
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
                range, num_points // num_threads, "Solving CVRP Using LKH", show_time
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
        
        # show time
        timer.end()
        timer.show_time()
        
        return self.tours    

    def __str__(self) -> str:
        return "CVRPLKHSolver"