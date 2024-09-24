import ctypes
import numpy as np
from .c_atsp_2opt import c_atsp_2opt_local_search

    
def atsp_2opt_local_search(
    init_tours: np.ndarray,
    dists: np.ndarray, 
    max_iterations_2opt: int = 5000
) -> np.ndarray:
    # prepare for decoding
    nodes_num = dists.shape[-1]
    init_tours = init_tours.astype(np.int16)
    dists = dists.astype(np.float32)
    tours = list()

    # check the number of dimension
    if init_tours.ndim == 1:
        init_tours = np.expand_dims(init_tours, axis=0)
    if init_tours.ndim != 2:
        raise ValueError("``init_tours`` must be a 1D or 2D array.")
    if dists.ndim == 2:
        dists = np.expand_dims(dists, axis=0)
    if dists.ndim != 3:
        raise ValueError("``dists`` must be a 2D or 3D array.")

    # tsp_mcts_decoder
    for idx in range(dists.shape[0]):
        # reshape to 1D
        _dists: np.ndarray = dists[idx] 
        _dists = _dists.reshape(-1)         

        # real decoding
        init_tour: np.ndarray = init_tours[idx]
        mcts_tour = c_atsp_2opt_local_search(
            init_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
            _dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
            nodes_num,
            max_iterations_2opt,
        )
        mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
        mcts_tour = np.append(mcts_tour, 0)
        tours.append(mcts_tour)

    # check shape
    tours = np.array(tours)
    if tours.shape[0] == 1:
        tours = tours[0]

    return tours