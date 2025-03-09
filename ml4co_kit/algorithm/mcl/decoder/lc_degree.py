import copy
import numpy as np


def mcl_lc_degree_decoder(graph: np.ndarray) -> np.ndarray:
    # preparation
    np.fill_diagonal(graph, 1)
    lc_graph = copy.deepcopy(graph)
    degrees: np.ndarray = lc_graph.sum(1)
    sol = np.zeros_like(degrees).astype(np.bool_)
    mask = np.zeros_like(degrees).astype(np.bool_)
    
    # local construction degree decoding
    while not mask.all():
        next_node = np.argmax(degrees)
        unconnect_nodes = np.where(lc_graph[next_node] == 0)[0]
        sol[unconnect_nodes] = False
        sol[next_node] = True
        mask[unconnect_nodes] = True
        mask[next_node] = True
        lc_graph[unconnect_nodes, :] = 0
        lc_graph[:, unconnect_nodes] = 0
        degrees = lc_graph.sum(1)
        degrees[mask==True] = 0
                
    # return
    return sol.astype(np.int32)