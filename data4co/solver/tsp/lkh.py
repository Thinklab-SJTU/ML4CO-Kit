import time
import numpy as np
import tsplib95
import pathlib
from tqdm import tqdm
from .base import TSPSolver
from .lkh_solver import lkh_solve


class TSPLKHSolver(TSPSolver):
    def __init__(
        self, 
        lkh_max_trials: int=1000,
        lkh_path: pathlib.Path="LKH",
        lkh_scale: int=1e6,
        lkh_runs: int=10,
        edge_weight_type: str="EUC_2D"
    ):
        """
        TSPLKHSolver
        Args:
            lkh_max_trials (int, optional): The maximum number of trials for 
                the LKH solver. Defaults to 1000.
            lkh_path (pathlib.Path, optional): The path to the LKH solver. 
                Defaults to "LKH".
            lkh_scale (int, optional): The scale factor for coordinates in the 
                LKH solver. Defaults to 1e6.
            lkh_runs (int, optional): The number of runs for the LKH solver. 
                Defaults to 10.
            edge_weight_type (str, optional):
                egde weights type of TSP, support ``EXPLICIT``, ``EUC_2D``, ``EUC_3D``,
                ``MAX_2D``, ``MAN_2D``, ``GEO``, ``GEOM``, ``ATT``, ``CEIL_2D``,
                ``CEIL_2D``, ``DSJRAND``
        """
        super(TSPLKHSolver, self).__init__()
        self.solver_type = "lkh"
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_scale = lkh_scale
        self.lkh_runs = lkh_runs
        self.edge_weight_type = edge_weight_type

    def _solve(self, nodes_coord: np.ndarray) -> list:
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'
        problem.type = 'TSP'
        problem.dimension = self.nodes_num
        problem.edge_weight_type = self.edge_weight_type
        problem.node_coords = {
            n + 1: nodes_coord[n] * self.lkh_scale for n in range(self.nodes_num)
        }
        solution = lkh_solve(
            solver=self.lkh_path, 
            problem=problem, 
            max_trials=self.lkh_max_trials, 
            runs=self.lkh_runs
        )
        tour = [n - 1 for n in solution[0]]   
        return tour
        
    def solve(
        self, 
        points: np.ndarray=None, 
        show_time: bool=False
    ) -> np.ndarray:
        start_time = time.time()
        if points is not None:
            self.from_data(points)
        if self.points is None:
            raise ValueError("points is None!")
        tours = []
        num_points = self.points.shape[0]
        if show_time:
            for idx in tqdm(range(num_points), desc="Solving TSP Using LKH"):
                tours.append(self._solve(self.points[idx]))
        else:
            for idx in range(num_points):
                tours.append(self._solve(self.points[idx]))
        self.tours = np.array(tours)
        zeros = np.zeros((self.tours.shape[0], 1))
        self.tours = np.append(self.tours, zeros, axis=1).astype(np.int32)
        if self.tours.ndim == 2 and self.tours.shape[0] == 1:
            self.tours = self.tours[0]
        end_time = time.time()
        if show_time:
            print(f"Use Time: {end_time - start_time}")
        return self.tours
    
    def regret_solve(
        self,
        points: np.ndarray,
        fixed_edges: tuple 
    ):
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'
        problem.type = 'TSP'
        problem.dimension = points.shape[0]
        problem.edge_weight_type = 'EUC_2D'
        problem.node_coords = {n + 1: self.lkh_scale * points[n] for n in range(points.shape[0])}
        problem.fixed_edges = [[n + 1 for n in fixed_edges]]
        solution = lkh_solve(
            solver=self.lkh_path, 
            problem=problem, 
            max_trials=self.lkh_max_trials, 
            runs=self.lkh_runs
        )
        tour = [n - 1 for n in solution[0]] + [0]
        np_tour = np.array(tour)
        return np_tour