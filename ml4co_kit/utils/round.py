import numpy as np
from typing import Callable


_RoundingFunc = Callable[[np.ndarray], np.ndarray]


def round_nearest(vals: np.ndarray):
    return np.round(vals).astype(np.int64)


def truncate(vals: np.ndarray):
    return vals.astype(np.int64)


def dimacs(vals: np.ndarray):
    return (10 * vals).astype(np.int64)


def exact(vals: np.ndarray):
    return round_nearest(1_000 * vals)


def no_rounding(vals):
    return vals


ROUND_FUNCS: dict[str, _RoundingFunc] = {
    "round": round_nearest,
    "trunc": truncate,
    "dimacs": dimacs,
    "exact": exact,
    "none": no_rounding,
}
