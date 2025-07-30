import math
import numpy as np
from typing import Union
from pyvrp.read import ROUND_FUNCS
from ml4co_kit.utils.type_utils import to_numpy
from ml4co_kit.utils.distance_utils import geographical

SUPPORT_NORM_TYPE = ["EUC_2D", "GEO"]


class OPChecker(object):
    def __init__(
        self,
        depots: Union[list, np.ndarray],
        points: Union[list, np.ndarray],
        norm: str = "EUC_2D"
    ):
        # depots
        depots = to_numpy(depots)
        if depots.ndim == 2 and depots.shape[0] == 1:
            depots = depots[0]
        if depots.ndim != 1:
            raise ValueError("depots must be 1D array.")
        
        # points
        points = to_numpy(points)
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

    def set_norm(self, norm: str):
        if norm not in SUPPORT_NORM_TYPE:
            message = (
                f"The norm type ({norm}) is not a valid type, "
                f"only {SUPPORT_NORM_TYPE} are supported."
            )
            raise ValueError(message)
        self.norm = norm

    def get_weight(self, x: np.ndarray, y: np.ndarray):
        if self.norm == "EUC_2D":
            return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
        elif self.norm == "GEO":
            return geographical(x, y)

    def calc_length(
        self, route: Union[np.ndarray, list], 
        to_int: bool = False, round_func: str="round" 
    ):
        if not to_int:
            round_func = "none"
        
        if (key := str(round_func)) in ROUND_FUNCS:
            round_func = ROUND_FUNCS[key]
        
        if not callable(round_func):
            raise TypeError(
                f"round_func = {round_func} is not understood. Can be a function,"
                f" or one of {ROUND_FUNCS.keys()}."
            )
        
        # length of route
        route_length = 0
        for i in range(len(route) - 1):
            length = self.get_weight(
                self.points[route[i]], self.points[route[i + 1]]
            )
            route_length += round_func(length)
            
        return route_length
    
    
class OPEvaluator(object):
    def __init__(self, prizes: Union[list, np.ndarray],):
        prizes = to_numpy(prizes)
        if prizes.ndim == 2 and prizes.shape[0] == 1:
            prizes = prizes[0]
        if prizes.ndim != 1:
            raise ValueError("prizes must be 1D array.")   
        self.prizes = prizes     
    
    def evaluate(self, route: Union[np.ndarray, list]):
        return self.prizes[to_numpy(route)[1:-1] - 1].sum()