import numpy as np
from typing import Union
from ml4co_kit.evaluate.tsp.base import TSPEvaluator


class CVRPEvaluator(TSPEvaluator):
    def __init__(
        self,
        depots: Union[list, np.ndarray],
        points: Union[list, np.ndarray], 
        norm: str = "EUC_2D"
    ):
        # depots
        if type(depots) == list:
            depots = np.array(depots)
        if depots.ndim == 2 and depots.shape[0] == 1:
            depots = depots[0]
        if depots.ndim != 1:
            raise ValueError("depots must be 1D array.")
        
        # points
        if type(points) == list:
            points = np.array(points)
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if points.ndim != 2:
            raise ValueError("points must be 2D array.")
        
        points_shape = points.shape
        coords = np.zeros(shape=(points_shape[0] + 1, points_shape[1]))
        coords[0] = depots
        coords[1:] = points
        self.points = coords
        self.set_norm(norm)