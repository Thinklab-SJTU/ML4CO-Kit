import ctypes
import numpy as np
from ml4co_kit.algorithm.tsp.decoder.c_tsp_mcts import c_mcts_decoder

    
def tsp_mcts_decoder(
    heatmap: np.ndarray, 
    points: np.ndarray, 
    time_limit: float,
    max_depth: int = 10, 
    type_2opt: int = 1, 
    max_iterations_2opt: int = 5000
) -> np.ndarray:
    # prepare for decoding
    nodes_num = heatmap.shape[-1]
    heatmap = heatmap.astype(np.float32)
    points = points.astype(np.float32)
    tours = list()

    # check the number of dimension
    if heatmap.ndim == 2:
        heatmap = np.expand_dims(heatmap, axis=0)
    if heatmap.ndim != 3:
        raise ValueError("``heatmap`` must be a 2D or 3D array.")
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)
    if points.ndim != 3:
        raise ValueError("``points`` must be a 2D or 3D array.")

    # tsp_mcts_decoder
    for idx in range(heatmap.shape[0]):
        _heatmap: np.ndarray = heatmap[idx]
        _points: np.ndarray = points[idx]

        # reshape to 1D
        _points = _points.reshape(-1)
        _heatmap = _heatmap.reshape(-1)            

        # real decoding
        tour = c_mcts_decoder(
            _heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
            _points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
            nodes_num,
            max_depth,
            ctypes.c_float(time_limit),
            type_2opt,
            max_iterations_2opt,
        )
        tour = np.ctypeslib.as_array(tour, shape=(nodes_num,))
        tour = np.append(tour, 0)
        tours.append(tour)

    # check shape
    tours = np.array(tours)
    if tours.shape[0] == 1:
        tours = tours[0]

    return tours