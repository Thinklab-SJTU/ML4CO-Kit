import numpy as np


def mcl_greedy_decoder(
    heatmap: np.ndarray, graph: np.ndarray
) -> np.ndarray:
    # preparation
    sol = np.zeros_like(heatmap).astype(np.bool_)
    mask = np.zeros_like(heatmap).astype(np.bool_)
    sorted_nodes = np.argsort(-heatmap)
    
    # greedy decoding
    for node in sorted_nodes:
        if not mask[node]:
            if (graph[node][sol]).all():
                unconnect_nodes = np.where(graph[node] == 0)[0]
                sol[unconnect_nodes] = False
                sol[node] = True
                mask[unconnect_nodes] = True
                mask[node] = True
    
    # return
    return sol.astype(np.int32)