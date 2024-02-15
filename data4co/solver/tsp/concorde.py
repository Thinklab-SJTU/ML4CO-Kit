import time
import numpy as np
from tqdm import tqdm
from .pyconcorde import TSPConSolver
from .base import TSPSolver


class TSPConcordeSolver(TSPSolver):
    def __init__(
        self, 
        concorde_scale: int=1e6, 
        edge_weight_type: str="EUC_2D"
    ):
        """
        TSPConcordeSolver
        Args:
            concorde_scale (int, optional): 
                The scale factor for coordinates in the Concorde solver.
            edge_weight_type (str, optional):
                egde weights type of TSP, support ``EXPLICIT``, ``EUC_2D``, ``EUC_3D``,
                ``MAX_2D``, ``MAN_2D``, ``GEO``, ``GEOM``, ``ATT``, ``CEIL_2D``,
                ``CEIL_2D``, ``DSJRAND``
        """
        super(TSPConcordeSolver, self).__init__()
        self.solver_type = "concorde"
        self.concorde_scale = concorde_scale
        self.norm = edge_weight_type

    def _solve(self, nodes_coord: np.ndarray) -> np.ndarray:
        solver = TSPConSolver.from_data(nodes_coord[:, 0] * self.concorde_scale, 
        nodes_coord[:, 1] * self.concorde_scale, norm=self.norm)
        solution = solver.solve(verbose=False)
        tour = solution.tour
        return tour
        
    def solve(
        self, 
        points: np.ndarray=None, 
        show_time: bool=False
    ) -> np.ndarray:
        start_time = time.time()
        # points
        if points is not None:
            self.from_data(points)
        if self.points is None:
            raise ValueError("points is None!")
        # solve
        tours = list()
        num_points = self.points.shape[0]
        if show_time:
            for idx in tqdm(range(num_points), desc="Solving TSP Using Concorde"):
                tours.append(self._solve(self.points[idx]))
        else:
            for idx in range(num_points):
                tours.append(self._solve(self.points[idx]))
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
