r"""
This module provides a class TSPLKHSolver for solving the TSP
using the LKH (Lin-Kernighan Heuristic) algorithm.
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


import pathlib
import numpy as np
from typing import Union
from multiprocessing import Pool
from ml4co_kit.utils import tsplib95
from ml4co_kit.solver.tsp.base import TSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.solver.tsp.lkh_solver import lkh_solve
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class TSPLKHSolver(TSPSolver):
    r"""
    The solver of TSP with the LKH algorithm.

    :param lkh_max_trials (int, optional): The maximum number of trials for the LKH solver. Defaults to 500.
    :param lkh_path (pathlib.Path, optional): The path to the LKH solver. Defaults to "LKH".
    :param scale (int, optional): The scale factor for coordinates in the LKH solver. Defaults to 1e6.
    :param lkh_runs (int, optional): The number of runs for the LKH solver. Defaults to 1.
    """
    def __init__(
        self,
        scale: int = 1e6,
        lkh_max_trials: int = 500,
        lkh_path: pathlib.Path = "LKH",
        lkh_runs: int = 1,
        lkh_seed: int = 1234,
        lkh_special: bool = False
    ):
        super(TSPLKHSolver, self).__init__(
            solver_type=SOLVER_TYPE.LKH, scale=scale
        )
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_runs = lkh_runs
        self.lkh_seed = lkh_seed
        self.lkh_special = lkh_special

    def _solve(self, nodes_coord: np.ndarray) -> list:
        r"""
        solve a single TSP instance using LKHSolver
        """
        problem = tsplib95.models.StandardProblem()
        problem.name = "TSP"
        problem.type = "TSP"
        problem.dimension = self.nodes_num
        problem.edge_weight_type = self.norm
        problem.node_coords = {
            n + 1: nodes_coord[n] * self.scale for n in range(self.nodes_num)
        }
        solution = lkh_solve(
            solver=self.lkh_path,
            problem=problem,
            max_trials=self.lkh_max_trials,
            runs=self.lkh_runs,
            seed=self.lkh_seed,
            special=self.lkh_special
        )
        tour = [n - 1 for n in solution[0]]
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
        Solve TSP using LKH algorithm with options for normalization,
        threading, and timing.

        :param points: np.ndarry or list, the points to solve TSP for. Defaults to None.
        :param norm: string, The norm to use for distance calculation. Defaults to "EUC_2D".
        :param normalize: boolean, whether to normalize the points. Defaults to "False".
        :param num_threads: int, The number of threads to use for solving. Defaults to 1.
        :param show_time: boolean, whether to show the time taken to solve. Defaults to "False".

        .. dropdown:: Example

            ::
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

    def regret_solve(
        self, points: np.ndarray, fixed_edges: tuple, norm: str = "EUC_2D"
    ):
        r"""   
        Sovle TSP with fixed edges using LKH algorithm.

        :param points: np.ndarry or list, the points to solve TSP for. Defaults to None.
        :param fixed_edges: tuple, The edges that must be included in the tour.
        :param norm: string, The norm to use for distance calculation. Defaults to "EUC_2D".

        .. dropdown:: Example

            ::
        """
        problem = tsplib95.models.StandardProblem()
        problem.name = "TSP"
        problem.type = "TSP"
        problem.dimension = points.shape[0]
        problem.edge_weight_type = norm
        problem.node_coords = {
            n + 1: self.scale * points[n] for n in range(points.shape[0])
        }
        problem.fixed_edges = [[n + 1 for n in fixed_edges]]
        solution = lkh_solve(
            solver=self.lkh_path,
            problem=problem,
            max_trials=self.lkh_max_trials,
            runs=self.lkh_runs,
        )
        tour = [n - 1 for n in solution[0]] + [0]
        np_tour = np.array(tour)
        return np_tour

    def __str__(self) -> str:
        return "TSPLKHSolver"