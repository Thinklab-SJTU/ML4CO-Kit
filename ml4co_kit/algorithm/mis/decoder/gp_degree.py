import numpy as np


def mis_gp_degree_decoder(graph: np.ndarray) -> np.ndarray:
    # preparation
    np.fill_diagonal(graph, 1)
    degrees: np.ndarray = graph.sum(1)
    sol = np.zeros_like(degrees).astype(np.bool_)
    mask = np.zeros_like(degrees).astype(np.bool_)
    sorted_nodes = np.argsort(degrees)

    # global prediction degree decoding
    for node in sorted_nodes:
        if not mask[node]:
            if (graph[node][sol]).sum() == 0:
                connect_nodes = np.where(graph[node] == 1)[0]
                sol[connect_nodes] = False
                sol[node] = True
                mask[connect_nodes] = True
                mask[node] = True

    # return
    return sol.astype(np.int32)