import ctypes
import numpy as np
from ml4co_kit.algorithm.tsp.decoder.c_tsp_insertion import c_insertion


def tsp_insertion_decoder(points: np.ndarray) -> np.ndarray:
    # prepare for decoding
    nodes_num = points.shape[-2]
    points = points.astype(np.float32)
    tours = list()
    
    # check the number of dimension
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)
    if points.ndim != 3:
        raise ValueError("``points`` must be a 2D or 3D array.")
    
    # tsp_insertion_decoder
    for idx in range(points.shape[0]):
        # random index
        index = np.arange(1, nodes_num)
        np.random.shuffle(index)
        random_index = np.insert(index, [0, len(index)], [0, 0])
        random_index = random_index.astype(np.int16)
        
        # greedy insertion
        _points: np.ndarray = points[idx]
        insertion_tour = c_insertion(
            random_index.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
            _points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nodes_num
        )
        insertion_tour = np.ctypeslib.as_array(insertion_tour, shape=(nodes_num+1,))
        tours.append(insertion_tour)
          
    # check shape
    tours = np.array(tours)
    if tours.shape[0] == 1:
        tours = tours[0]
    return tours