import numpy as np
from typing import Union
import ctypes
from numpy.ctypeslib import ndpointer
import sys
import subprocess

def to_numpy(
    x: Union[np.ndarray, list]
) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)

import os
file_dir = os.path.dirname(os.path.abspath(__file__))
arw_path = os.path.join(file_dir, "arw/KaMIS/deploy/libarw_1iter.so")

if not os.path.exists(arw_path):
    recompile_script = os.path.join(file_dir, "arw", "recompile.py")
    subprocess.run([sys.executable, recompile_script], check=True)

lib = ctypes.CDLL(arw_path)

lib.arw_1iter.restype = None
lib.arw_1iter.argtypes = [
    ctypes.c_int,                                     # num_nodes
    ctypes.c_int,                                     # edge_count
    ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # xadj
    ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # adjncy
    ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # initial_solution
    ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")   # output
]

def mis_arw_local_search_1iter(xadj: np.ndarray, adjncy: np.ndarray, ini_sol: np.ndarray):
    
    num_nodes = xadj.shape[0] - 1
    num_edges = int(adjncy.shape[0] / 2)
    ini_sol = to_numpy(ini_sol)
    
    xadj = np.ascontiguousarray(xadj, dtype=np.int32)
    ini_sol = np.ascontiguousarray(ini_sol, dtype=np.int32)
    adjncy = np.ascontiguousarray(adjncy, dtype=np.int32)
    output = np.empty(num_nodes, dtype=np.int32)
    
    lib.arw_1iter(
        num_nodes,
        num_edges,
        xadj,
        adjncy,
        ini_sol,
        output
    )
    return output