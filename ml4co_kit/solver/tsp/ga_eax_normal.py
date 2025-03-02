r"""
GA-EAX Solver for solving TSPs.

The GA-EAX (Genetic Algorithm with EAX) is a hybrid algorithm that combines 
Genetic Algorithms (GA) with the EAX (Edge Assembly Crossover) operator. 
This hybrid approach is designed to solve the TSP more efficiently by incorporating 
the strengths of both genetic algorithms and EAX. EAX is an advanced crossover operator 
used in evolutionary algorithms for solving TSP and other combinatorial problems.

We follow https://github.com/nagata-yuichi/GA-EAX/tree/main for the implementation of GA-EAX Solver. 
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
from ml4co_kit.solver.tsp.base import TSPSolver
from ml4co_kit.solver.tsp.c_ga_eax_normal import (
    GA_EAX_NORMAL_TMP_PATH, tsp_ga_eax_normal_solve
)
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.evaluate.tsp.base import TSPEvaluator
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class TSPGAEAXSolver(TSPSolver):
    r"""
    Solve TSPs using GA-EAX solver.

    :param scale: int, the scaling factor for the coordinates of the nodes.
    :param max_trials: int, Tthe maximum number of trials for the genetic algorithm.
    :param population_num: int, the number of individuals in the population.
    :param offsping_num: int, the number of offspring produced in each generation.
    :param show_info: boolean, whether to display the information during the solving process.
    """
    def __init__(
        self,
        scale: int = 1e5,
        max_trials: int = 1,
        population_num: int = 100,
        offspring_num: int = 30,
        show_info: bool = False
    ):
        super(TSPGAEAXSolver, self).__init__(
            solver_type=SOLVER_TYPE.GA_EAX, scale=scale
        )
        self.max_trials = max_trials
        self.population_num = population_num
        self.offspring_num = offspring_num
        self.show_info = show_info
        
    def _read_solution(self, file_path: str) -> np.ndarray:
        r"""
        Read solutions from a file.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        tour_list = list()
        for idx in range(len(lines) // 2):
            tour_str = lines[idx * 2 + 1]
            tour_split = tour_str.split(" ")[:-1]
            tour = [int(node) - 1 for node in  tour_split]
            tour.append(0)
            tour_list.append(tour)
        return np.array(tour_list)
    
    def _solve(self, nodes_coord: np.ndarray) -> list:
        r"""
        solve a single TSP problem.
        """
        # eval
        eval = TSPEvaluator(nodes_coord)
        
        # scale
        nodes_coord = (nodes_coord * self.scale).astype(np.int64)
        
        # generate .tsp file
        name = uuid.uuid4().hex[:9]
        tmp_solver = TSPSolver()
        tmp_solver.from_data(points=nodes_coord, ref=False)
        tmp_solver.to_tsplib_folder(
            tsp_save_dir=GA_EAX_NORMAL_TMP_PATH, tsp_filename=name
        )
        
        # Intermediate files
        tsp_abs_path = os.path.join(GA_EAX_NORMAL_TMP_PATH, f"{name}.tsp")
        sol_abs_path_1 = os.path.join(GA_EAX_NORMAL_TMP_PATH, f"{name}_BestSol")
        sol_abs_path_2 = os.path.join(GA_EAX_NORMAL_TMP_PATH, f"{name}_Result")
        
        # solve
        tsp_ga_eax_normal_solve(
            max_trials=self.max_trials, sol_name=name, 
            population_num=self.population_num,
            offspring_num=self.offspring_num, 
            tsp_name=tsp_abs_path, show_info=self.show_info
        )
        
        # read data from .sol
        tours = self._read_solution(sol_abs_path_1)
        costs = np.array([eval.evaluate(tour) for tour in tours])
        min_cost_idx = np.argmin(costs)
        best_tour = tours[min_cost_idx].tolist()
        
        # clear files
        intermediate_files = [tsp_abs_path, sol_abs_path_1, sol_abs_path_2]
        for file_path in intermediate_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        return best_tour

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
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return self.tours
    
    def __str__(self) -> str:
        return "TSPGAEAXSolver"