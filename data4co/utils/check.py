import numpy as np


def check_dim(array: np.ndarray, dim: int):
    if array.ndim != dim:
        raise ValueError(f"Dimension mismatch!, should be {dim}D but find {array.ndim}D")
        
        