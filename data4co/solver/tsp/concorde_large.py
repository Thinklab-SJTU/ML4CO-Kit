import os
import time
import uuid
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from data4co.utils import check_dim
from .concorde import TSPConcordeSolver


class TSPConcordeLargeSolver(TSPConcordeSolver):
    def __init__(
        self, 
        concorde_scale: int=1e6, 
        edge_weight_type: str="EUC_2D"
    ):
        """
        TSPLargeConcordeSolver
        Args:
            concorde_scale (int, optional): 
                The scale factor for coordinates in the Concorde solver.
            edge_weight_type (str, optional):
                egde weights type of TSP, support ``EXPLICIT``, ``EUC_2D``, ``EUC_3D``,
                ``MAX_2D``, ``MAN_2D``, ``GEO``, ``GEOM``, ``ATT``, ``CEIL_2D``,
                ``CEIL_2D``, ``DSJRAND``
        """
        super(TSPConcordeLargeSolver, self).__init__(
            concorde_scale=concorde_scale,
            edge_weight_type=edge_weight_type
        )
        self.solver_type = "concorde-large"
    
    def solve(
        self, 
        points: np.ndarray=None, 
        num_threads: int=1,
        max_time: float=600,
        show_time: bool=False
    ) -> np.ndarray:
        start_time = time.time()
        # points
        if points is not None:
            self.from_data(points)
        if self.points is None:
            raise ValueError("points is None!")
        check_dim(self.points, 3)

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
        self.tours = np.array(tours)
        zeros = np.zeros((self.tours.shape[0], 1))
        self.tours = np.append(self.tours, zeros, axis=1).astype(np.int32)
        if self.tours.ndim == 2 and self.tours.shape[0] == 1:
            self.tours = self.tours[0]
        end_time = time.time()
        if show_time:
            print(f"Use Time: {end_time - start_time}")
        return self.tours

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
