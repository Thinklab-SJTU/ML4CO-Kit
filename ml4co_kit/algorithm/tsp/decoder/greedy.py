import numpy as np
from ml4co_kit.algorithm.tsp.decoder.cython_tsp_greedy import cython_tsp_greedy


def tsp_greedy_decoder(heatmap: np.ndarray) -> np.ndarray:
    # prepare for decoding
    heatmap = heatmap.astype("double")
    tours = list()
    
    # check the number of dimension
    dim_2 = False
    if heatmap.ndim == 2:
        dim_2 = True
        heatmap = np.expand_dims(heatmap, axis=0)
    if heatmap.ndim != 3:
        raise ValueError("``heatmap`` must be a 2D or 3D array.")
    
    # tsp_greedy_decoder
    for idx in range(heatmap.shape[0]):
        adj_mat = cython_tsp_greedy(heatmap[idx])[0]
        adj_mat = np.asarray(adj_mat)
        tour = [0]
        cur_node = 0
        cur_idx = 0
        while(len(tour) < adj_mat.shape[0] + 1):
            cur_idx += 1
            cur_node = np.nonzero(adj_mat[cur_node])[0]
            if cur_idx == 1:
                cur_node = cur_node.max()
            else:
                cur_node = cur_node[1] if cur_node[0] == tour[-2] else cur_node[0]
            tour.append(cur_node)
        tours.append(tour)
            
    # check shape
    tours = np.array(tours)
    if dim_2:
        tours = tours[0]
    return tours