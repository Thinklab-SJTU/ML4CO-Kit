import numpy as np


def mvc_gp_degree_decoder(graph: np.ndarray) -> np.ndarray:
    # preparation
    np.fill_diagonal(graph, 0)
    degrees: np.ndarray = graph.sum(1)
    sol = np.zeros_like(degrees).astype(np.bool_)
    mask = np.zeros_like(degrees).astype(np.bool_)
    sorted_nodes = np.argsort(degrees)

    # global prediction degree decoding
    for node in sorted_nodes:
        if not mask[node]:
            connect_nodes = np.where(graph[node] == 1)[0]
            sol[connect_nodes] = True
            sol[node] = False
            mask[connect_nodes] = True
            mask[node] = True

    # return
    return sol.astype(np.int32)