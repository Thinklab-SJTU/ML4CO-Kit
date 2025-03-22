import ctypes
import numpy as np
from typing import Union
from ml4co_kit.utils.type_utils import to_numpy
from ml4co_kit.algorithm.cvrp.local_search.c_classic import c_cvrp_local_search


def cvrp_classic_local_search(
    init_tour: Union[np.ndarray, list], 
    depot: Union[np.ndarray, list],
    points: Union[np.ndarray, list], 
    demands: Union[np.ndarray, list], 
    coords_scale: int = 1000,
    demands_scale: int = 1000,
    seed: int = 1234,
) -> np.ndarray:
    # type
    init_tour = to_numpy(init_tour)
    depot = to_numpy(depot)
    points = to_numpy(points)
    demands = to_numpy(demands)
    
    # check
    if init_tour.ndim != 1:
        raise ValueError("``init_tour`` must be a 1D array.")
    if depot.ndim != 1:
        raise ValueError("``depot`` must be a 1D array.")
    if points.ndim != 2:
        raise ValueError("``points`` must be a 2D array.")
    if demands.ndim != 1:
        raise ValueError("``demands`` must be a 2D array.")
    if init_tour[0] != 0 or init_tour[-1] > 0:
        raise ValueError("Illegal initial solution!")
    if len(demands) == len(points):
        demands = np.insert(demands, 0, 0)
        
    # prepare for local search
    nodes_num = points.shape[0] + 1
    input_init_tour = init_tour.astype(np.int16)
    coords = np.concatenate([np.expand_dims(depot, 0), points], axis=0)
    coords = coords.astype(np.float32).reshape(-1)
    demands = demands.astype(np.float32).reshape(-1)
    
    # local search
    ls_tour = c_cvrp_local_search(
        input_init_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
        demands.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        nodes_num,
        len(input_init_tour),
        coords_scale,
        demands_scale,
        seed
    )
    
    # format
    ls_tour = np.ctypeslib.as_array(ls_tour, shape=(len(input_init_tour)+2,))
    if ls_tour[0] == -1:
        return init_tour
    else:
        return ls_tour[:np.where(ls_tour==-1)[0][0]]