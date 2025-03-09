import numpy as np


def mis_beam_decoder(
    heatmap: np.ndarray, graph: np.ndarray, beam_size: int
) -> np.ndarray:
    # preparation
    np.fill_diagonal(graph, 1)
    empty_flag = [True for _ in range(beam_size)]
    independent_set = [list() for _ in range(beam_size)] 
    sol = np.zeros_like(heatmap)
    beam_sols = np.repeat(sol.reshape(1, -1), beam_size, axis=0)
    mask = np.zeros_like(heatmap).astype(np.bool_)
    sorted_nodes = np.argsort(-heatmap)
    
    # beam decoding
    for node in sorted_nodes:
        if not mask[node]:
            for idx in range(beam_size):
                if empty_flag[idx]:
                    independent_set[idx].append(node)
                    beam_sols[idx][node] = 1
                    empty_flag[idx] = False
                    break
                if (graph[node][independent_set[idx]] == 0).all():
                    independent_set[idx].append(node)
                    beam_sols[idx][node] = 1
                    break
                    
    # select best
    best_idx = np.argmax(beam_sols.sum(axis=1))

    # return
    return beam_sols[best_idx]