import numpy as np


def tsp_greedy(
    adj_mat: np.ndarray, 
    np_points: np.ndarray, 
    parallel_sampling: int=1, 
    device: str="cpu", 
    **kwargs
):
    raise NotImplementedError


def tsp_2opt(
    np_points: np.ndarray,
    tours: np.ndarray,
    adj_mat: np.ndarray = None,
    device="cpu",
    **kwargs
):
    raise NotImplementedError
