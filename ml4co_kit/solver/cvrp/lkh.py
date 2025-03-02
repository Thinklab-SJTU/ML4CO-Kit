r"""
LKH Solver for solving CVRPs.

LKH is a heuristic algorithm that uses k-opt move strategies
to find approximate optimal solutions to problems.
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
    r"""
    Solve CVRPs using LKH solver.
    
    :param depots_scale: int, the scale of the depots.
    :param points_scale: int, the scale of the customer points.
    :param demands_scale: int, the scale of the demands of customer points.
    :param capacities_scale: int, the scale of the capacities of the car.
    :param lkh_max_trials: int, The maximum number of trials for the LKH solver.
    :param lkh_path: string, The path to the LKH solver.
    :param lkh_runs: int, The number of runs for the LKH solver.
    :param lkh_seed, int, the random number seed for the LKH solver.
    :param lkh_special, boolean, whether to solve in a special way.
    """
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

    def _write_parameter_file(
        self,
        save_path: str,
        vrp_file_path: str,
        tour_path: str
    ):
        r"""
        writing max_trials, runs, and seeds to problem_file, and writing tour_path to tour_file.
        
        :param save_path: string, the path to save the files.
        :param atsp_file_path: string, the path to save the problem_file.
        :param tour_path: string, the path to save the tour_file.
        """
        with open(save_path, "w") as f:
            f.write(f"PROBLEM_FILE = {vrp_file_path}\n")
            f.write(f"MAX_TRIALS = {self.lkh_max_trials}\n")
            if self.lkh_special:
                f.write("SPECIAL\n")
            f.write(f"RUNS = {self.lkh_runs}\n")
            f.write(f"SEED = {self.lkh_seed}\n")
            f.write(f"TOUR_FILE = {tour_path}\n")
    
    def _read_lkh_solution(self, tour_path: str) -> list:
        r"""
        read solutions in vrp format.
        """
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
        r"""
        solve a single CVRP instance using LKHSolver
        """
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
        self._write_parameter_file(
            save_path=para_save_path,
            vrp_file_path=vrp_save_path,
            tour_path=tour_save_path
        )
        
        # solve
        with open(log_save_path, "w") as f:
            check_call([self.lkh_path, para_save_path], stdout=f)
            
        # read solution
        tour = self._read_lkh_solution(tour_save_path)
        
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
        r"""
        Solve CVRP using LKH algorithm with options for normalization,
        threading, and timing.

        :param depots: np.ndarray, the depots coordinates data called by the solver during solving,
            they may initially be same as ``ori_depots``, but may later undergo standardization
            or scaling processing.
        :param points:  np.ndarray, the customer points coordinates data called by the solver
            during solving, they may initially be same as ``ori_depots``, but may later undergo
            standardization or scaling processing.
        :param demands: np.ndarray, the demands of each customer points.
        :param capacities: np.ndarray, the capacities of the car.
        :param norm: string, the norm used to calcuate the distance.
        :param normalize: boolean, whether to normalize the points. Defaults to "False".
        :param num_threads: int, The number of threads to use for solving. Defaults to 1.
        :param show_time: boolean, whether to show the time taken to solve. Defaults to "False".

        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import CVRPLKHSolver
                
                # create CVRPLKHSolver
                >>> solver = CVRPLKHSolver(lkh_max_trials=1)

                # load data and reference solutions from ``.vrp`` file
                >>> solver.from_vrplib(
                        vrp_file_path="examples/cvrp/vrplib_1/problem/A-n32-k5.vrp",
                        sol_file_path="examples/cvrp/vrplib_1/solution/A-n32-k5.sol",
                        ref=False,
                        norm="EUC_2D",
                        normalize=False
                    )
                    
                # solve
                >>> solver.solve()
                [[ 0,  6,  3,  2, 23, 14, 24,  0, 12,  1, 16, 30,  0,  7, 13, 17,
                19, 31, 21, 26,  0, 28,  4, 11,  8, 18,  9, 22, 27,  0, 29, 15,
                10, 25,  5, 20,  0]]
         """
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