import copy
import numpy as np


def mis_lc_degree_decoder(graph: np.ndarray) -> np.ndarray:
    # preparation
    np.fill_diagonal(graph, 1)
    lc_graph = copy.deepcopy(graph)
    degrees: np.ndarray = lc_graph.sum(1)
    sol = np.zeros_like(degrees).astype(np.bool_)
    mask = np.zeros_like(degrees).astype(np.bool_)
    
    # local construction degree decoding
    while not mask.all():
        next_node = np.argmin(degrees)
        connect_nodes = np.where(lc_graph[next_node] == 1)[0]
        sol[connect_nodes] = False
        sol[next_node] = True
        mask[connect_nodes] = True
        mask[next_node] = True
        lc_graph[connect_nodes, :] = 0
        lc_graph[:, connect_nodes] = 0
        lc_graph[next_node, :] = 0
        lc_graph[:, next_node] = 0
        degrees = lc_graph.sum(1)
        degrees[mask==True] = len(degrees)
                
    # return
    return sol.astype(np.int32)