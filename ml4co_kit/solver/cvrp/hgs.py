r"""
HGS solver for solving CVRPs. 

HGS is a metaheuristic algorithm that employs competitive survival 
strategies inspired by the dynamics of predator-prey interactions to 
find approximate optimal solutions to complex optimization problems. 

We follow https://github.com/vidalt/HGS-CVRP/tree/main/Program for the implementation of HGS.
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
import numpy as np
from typing import Union
from multiprocessing import Pool
from ml4co_kit.solver.cvrp.base import CVRPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer
from ml4co_kit.solver.cvrp.c_hgs import cvrp_hgs_solver, HGS_TMP_PATH


class CVRPHGSSolver(CVRPSolver):
    r"""
    Solve CVRPs using HGS solver.
    
    :param depots_scale: int, the scale of the depots.
    :param points_scale: int, the scale of the customer points.
    :param demands_scale: int, the scale of the demands of customer points.
    :param capacities_scale: int, the scale of the capacities of the car.
    :param time_limit: float, the limit of running time.
    :param show_info: boolean, whether to show detail information.
    """
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
            solver_type=SOLVER_TYPE.HGS, 
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
        r"""
        solve a single CVRP instance using HGSSolver
        """
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
        r"""
        Solve CVRP using HGS algorithm with options for normalization,
        threading, and timing

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
        """
        # preparation
        self.from_data(
            depots=depots, points=points, demands=demands,
            capacities=capacities, norm=norm, normalize=normalize
        )
        timer = Timer(apply=show_time)
        timer.start()

        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:   
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
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
                range, num_points // num_threads, self.solve_msg, show_time
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
        return "CVRPHGSSolver"