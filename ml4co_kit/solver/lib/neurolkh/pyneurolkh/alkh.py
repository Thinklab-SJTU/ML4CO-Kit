import ctypes
import numpy as np
from typing import Tuple
from ml4co_kit.solver.lib.neurolkh.pyneurolkh.c_alkh_v1 import c_alkh_tool_v1
from ml4co_kit.solver.lib.neurolkh.pyneurolkh.c_alkh_v2 import c_alkh_tool_v2


def neurolkh_alkh_tool_v1(
    points: np.ndarray, 
    penalty: np.ndarray, 
    in_candidates_num: int, 
    out_candidates_num: int, 
    scale: float, 
    lr: float,
    initial_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # prepare
    nodes_num = points.shape[0]
    points = points.astype(np.float32).reshape(-1)
    penalty = penalty / 100
    
    # solve
    results = c_alkh_tool_v1(
        nodes_num,
        points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        penalty.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        in_candidates_num,
        out_candidates_num,
        ctypes.c_float(scale),
        ctypes.c_float(lr),
        initial_period,
    )
    
    # results -> penalty, exp_nodes, candidates(new)
    results = np.ctypeslib.as_array(results, shape=((out_candidates_num + 2) * nodes_num,))
    exp_nodes = (results[0: nodes_num]).astype(np.int32)
    penalty = (results[nodes_num:2*nodes_num] / scale).astype(np.float32)
    candidates = (results[2*nodes_num:]).astype(np.int32)
    candidates = candidates.reshape(out_candidates_num, nodes_num).transpose(1, 0)

    return penalty*100, exp_nodes, candidates


def neurolkh_alkh_tool_v2(
    points: np.ndarray, 
    penalty: np.ndarray, 
    candidates: np.ndarray, 
    in_candidates_num: int, 
    out_candidates_num: int, 
    scale: float, 
    lr: float,
    initial_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # prepare
    nodes_num = points.shape[0]
    points = points.astype(np.float32).reshape(-1)
    penalty = penalty / 100
    
    # solve
    results = c_alkh_tool_v2(
        nodes_num,
        points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        penalty.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        candidates.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        in_candidates_num,
        out_candidates_num,
        ctypes.c_float(scale),
        ctypes.c_float(lr),
        initial_period,
    )
    
    # results -> penalty, exp_nodes, candidates(new)
    results = np.ctypeslib.as_array(results, shape=((out_candidates_num + 2) * nodes_num,))
    exp_nodes = (results[0: nodes_num]).astype(np.int32)
    penalty = (results[nodes_num:2*nodes_num] / scale).astype(np.float32)
    candidates = (results[2*nodes_num:]).astype(np.int32)
    candidates = candidates.reshape(out_candidates_num, nodes_num).transpose(1, 0)
    return penalty * 100, exp_nodes, candidates