import numpy as np


def tsp_greedy(adj_mat, np_points, parallel_sampling=1, device="cpu", **kwargs):
    raise NotImplementedError


def tsp_2opt(
    np_points: np.ndarray,
    tours: np.ndarray,
    adj_mat: np.ndarray = None,
    device="cpu",
    **kwargs
):
    raise NotImplementedError
