import math
import numpy as np
from typing import Union
from pyvrp.read import ROUND_FUNCS
from ml4co_kit.utils.distance_utils import geographical


SUPPORT_NORM_TYPE = ["EUC_2D", "GEO"]


class TSPEvaluator(object):
    def __init__(self, points: Union[list, np.ndarray], norm: str = "EUC_2D"):
        if type(points) == list:
            points = np.array(points)
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if points.ndim != 2:
            raise ValueError("points must be 2D array.")
        self.points = points
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

    def evaluate(
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
        
        total_cost = 0
        for i in range(len(route) - 1):
            cost = self.get_weight(self.points[route[i]], self.points[route[i + 1]])
            total_cost += round_func(cost)

        return total_cost