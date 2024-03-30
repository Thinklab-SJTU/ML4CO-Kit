import os
import time
import uuid
import numpy as np
from tqdm import tqdm
from typing import Union
from multiprocessing import Process
from .concorde import TSPConcordeSolver


class TSPConcordeLargeSolver(TSPConcordeSolver):
    def __init__(
        self, 
        scale: int=1e6, 
    ):
        """
        TSPLargeConcordeSolver
        Args:
            scale (int, optional): 
                The scale factor for coordinates in the Concorde solver.
        """
        super(TSPConcordeLargeSolver, self).__init__(
            scale=scale
        )
        self.solver_type = "concorde-large"
    
    def solve(
        self, 
        points: Union[np.ndarray, list]=None,
        norm: str="EUC_2D",
        normalize: bool=False,
        num_threads: int=1,
        max_time: float=600,
        show_time: bool=False
    ) -> np.ndarray:
        # prepare
        self.from_data(points, norm, normalize)
        start_time = time.time()

        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            if show_time:
                for idx in tqdm(range(num_points), desc="Solving TSP Using Concorde"):
                    name = uuid.uuid4().hex
                    filename = f"{name[0:9]}.sol"
                    proc = Process(target=self._solve, args=(self.points[idx], name))
                    proc.start()
                    start_time = time.time()
                    solve_finished = False
                    while(time.time() - start_time < max_time):
                        if os.path.exists(filename):
                            solve_finished = True
                            time.sleep(1)
                            break
                    proc.terminate()
                    proc.join(timeout=1)
                    if solve_finished:
                        tour = self.read_from_sol(filename)
                        tours.append(tour)
                        self.clear_tmp_files(name)
                    else:
                        self.clear_tmp_files(name)
                        raise TimeoutError()
            else:
                for idx in range(num_points):
                    name = uuid.uuid4().hex
                    filename = f"{name[0:9]}.sol"
                    proc = Process(target=self._solve, args=(self.points[idx], name))
                    proc.start()
                    start_time = time.time()
                    solve_finished = False
                    while(time.time() - start_time < max_time):
                        if os.path.exists(filename):
                            solve_finished = True
                            break
                    proc.terminate()
                    proc.join(timeout=1)
                    if solve_finished: 
                        tour = self.read_from_sol(filename)
                        tours.append(tour)
                        self.clear_tmp_files(name)
                    else:
                        self.clear_tmp_files(name)
                        raise TimeoutError()
        else:
            raise ValueError("TSPConcordeLargeSolver Only supports single threading!")

        # format
        tours = np.array(tours)
        zeros = np.zeros((tours.shape[0], 1))
        tours = np.append(tours, zeros, axis=1).astype(np.int32)
        if tours.ndim == 2 and tours.shape[0] == 1:
            tours = tours[0]
        self.read_tours(tours)
        end_time = time.time()
        if show_time:
            print(f"Use Time: {end_time - start_time}")
        return tours

    def read_from_sol(self, filename: str) -> np.ndarray:
        with open(filename, 'r') as file:
            gt_tour = list()
            first_line = True
            for line in file:
                if first_line:
                    first_line = False
                    continue
                line = line.strip().split(' ')
                for node in line:
                    gt_tour.append(int(node))
            gt_tour.append(0)
        return np.array(gt_tour)
