import os
import time
import uuid
import numpy as np
from typing import Union
from multiprocessing import Pool
from ml4co_kit.solver.tsp.base import TSPSolver
from ml4co_kit.solver.tsp.c_ga_eax_normal import (
    GA_EAX_NORMAL_TMP_PATH, tsp_ga_eax_normal_solve
)
from ml4co_kit.evaluate.tsp.base import TSPEvaluator
from ml4co_kit.utils.run_utils import iterative_execution


class TSPGAEAXSolver(TSPSolver):
    def __init__(
        self,
        scale: int = 1e5,
        max_trials: int = 10,
        population_num: int = 100,
        offspring_num: int = 30,
    ):
        super(TSPGAEAXSolver, self).__init__(solver_type="GA-EAX", scale=scale)
        self.max_trials = max_trials
        self.population_num = population_num
        self.offspring_num = offspring_num
    
    def read_solution(self, file_path: str) -> np.ndarray:
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
        # eval
        eval = TSPEvaluator(nodes_coord)
        
        # scale
        nodes_coord = (nodes_coord * self.scale).astype(np.int64)
        
        # generate .tsp file
        name = uuid.uuid4().hex[:9]
        tmp_solver = TSPSolver()
        tmp_solver.from_data(nodes_coord)
        tmp_solver.to_tsp(GA_EAX_NORMAL_TMP_PATH, filename=name)
        
        # Intermediate files
        tsp_abs_path = os.path.join(GA_EAX_NORMAL_TMP_PATH, f"{name}-0.tsp")
        sol_abs_path_1 = os.path.join(GA_EAX_NORMAL_TMP_PATH, f"{name}_BestSol")
        sol_abs_path_2 = os.path.join(GA_EAX_NORMAL_TMP_PATH, f"{name}_Result")
        
        # solve
        tsp_ga_eax_normal_solve(
            max_trials=self.max_trials, sol_name=name, population_num=self.population_num,
            offspring_num=self.offspring_num, tsp_name=f"{name}-0.tsp"
        )
        
        # read data from .sol
        tours = self.read_solution(sol_abs_path_1)
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
        # prepare
        self.from_data(points, norm, normalize)
        start_time = time.time()

        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(
                range, num_points, "Solving TSP Using GA-EAX", show_time
            ):
                tours.append(self._solve(self.points[idx]))
        else:
            batch_points = self.points.reshape(-1, num_threads, p_shape[-2], p_shape[-1])
            for idx in iterative_execution(
                range, num_points // num_threads, "Solving TSP Using GA-EAX", show_time
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
        if tours.ndim == 2 and tours.shape[0] == 1:
            tours = tours[0]
        self.read_tours(tours)
        end_time = time.time()
        if show_time:
            print(f"Use Time: {end_time - start_time}")
        return tours