import ctypes
import numpy as np
from ml4co_kit.algorithm.atsp.decoder.c_greedy import c_atsp_greedy_decoder


def atsp_greedy_decoder(heatmap: np.ndarray) -> np.ndarray:
    # prepare for decoding
    nodes_num = heatmap.shape[-1]
    tours = list()
    
    # check the number of dimension
    dim_2 = False
    if heatmap.ndim == 2:
        dim_2 = True
        heatmap = np.expand_dims(heatmap, axis=0)
    if heatmap.ndim != 3:
        raise ValueError("``heatmap`` must be a 2D or 3D array.")
    
    # atsp_greedy_decoder
    for idx in range(heatmap.shape[0]):
        _heatmap = heatmap[idx]
        tour = (ctypes.c_int * nodes_num)(*(list(range(nodes_num))))
        cost = ctypes.c_double(0)
        _heatmap = (ctypes.c_double *(nodes_num**2))(*_heatmap.reshape(nodes_num*nodes_num).tolist())
        c_atsp_greedy_decoder(nodes_num, _heatmap, tour, ctypes.byref(cost))
        tour = np.array(list(tour))
        tour = np.append(tour, tour[0])
        tours.append(tour)

    # check shape
    tours = np.array(tours)
    if dim_2:
        tours = tours[0]
    return tours