import math
import numpy as np
from typing import Union
from pyvrp.read import ROUND_FUNCS


class ATSPEvaluator(object):
    def __init__(self, dists: Union[list, np.ndarray]):
        if type(dists) == list:
            dists = np.array(dists)
        if dists.ndim == 3 and dists.shape[0] == 1:
            dists = dists[0]
        if dists.ndim != 2:
            raise ValueError("dists must be 2D array.")
        self.dists = dists

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
            cost = self.dists[route[i]][route[i + 1]]
            total_cost += round_func(cost)

        return total_cost