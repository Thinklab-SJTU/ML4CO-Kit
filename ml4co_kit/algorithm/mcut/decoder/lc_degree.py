import copy
import numpy as np


def mcut_lc_degree_decoder(graph: np.ndarray) -> np.ndarray:
    # preparation
    np.fill_diagonal(graph, 0)
    nodes_num = graph.shape[0]
    lc_graph = copy.deepcopy(graph)
    mask = np.zeros(shape=(nodes_num,)).astype(np.bool_)
    set_A = np.zeros(shape=(nodes_num,)).astype(np.bool_)
    set_B = np.zeros(shape=(nodes_num,)).astype(np.bool_)
    set_A[0] = True # default
    mask[0] = True # default
    
    # local construction degree decoding
    while not mask.all():
        # get degree
        degree_A = lc_graph[set_A].sum(0)
        degree_B = lc_graph[set_B].sum(0)
        degree_A[mask] = 0
        degree_B[mask] = 0
        
        # select next node and update
        max_A = np.max(degree_A)
        max_B = np.max(degree_B)
        if max_A > max_B:
            next_node = np.argmax(degree_A)
            set_B[next_node] = True
            mask[next_node] = True
        else:
            next_node = np.argmax(degree_B)
            set_A[next_node] = True
            mask[next_node] = True
                
    # return
    return set_A.astype(np.int32)