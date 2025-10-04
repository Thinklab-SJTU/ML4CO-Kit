import ctypes
import numpy as np
from typing import Tuple
from ml4co_kit.solver.lib.neurolkh.pyneurolkh.c_sparse_points import c_sparse_points


def neurolkh_sparser(
    points: np.ndarray, sparse_factor: int = 20, scale: float = 1e6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes_num = points.shape[0]
    results = c_sparse_points(
        nodes_num,
        points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sparse_factor,
        ctypes.c_float(scale),
    )
    results = np.ctypeslib.as_array(results, shape=(3 * nodes_num * sparse_factor,))
    results = results.reshape(nodes_num, 3*sparse_factor)
    edge_index = results[:, range(0, 3*sparse_factor, 3)]
    graph = results[:, range(1, 3*sparse_factor+1, 3)] / scale
    graph = graph.astype(np.float32)
    inverse_edge_index = results[:, range(2, 3*sparse_factor+2, 3)]
    return edge_index, graph, inverse_edge_index