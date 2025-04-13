import numpy as np


def cvrp_greedy_decoder(
    heatmap: np.ndarray, norm_demand: np.ndarray, cp_alpha: float = 0.5
) -> np.ndarray:
    # prepare for decoding
    np.fill_diagonal(heatmap, 0)
    current_capacity = 1.0
    nodes_visited = np.zeros_like(norm_demand).astype(np.bool_)
    tour = [0]
    current_node = 0
    nodes_visited[0] = True
    
    # greedy search
    while not nodes_visited.all():
        # find next node
        if current_capacity >= cp_alpha:
            next_node = np.argmax(heatmap[current_node][1:]) + 1
        else:
            next_node = np.argmax(heatmap[current_node])
            if norm_demand[next_node] > current_capacity:
                next_node = 0
        
        # update
        nodes_visited[next_node] = True
        tour.append(next_node)
        if next_node == 0:
            current_capacity = 1.0
        else:
            current_capacity -= norm_demand[next_node]
            heatmap[:, next_node] = -1
        current_node = next_node
    
    # return
    tour.append(0)
    return np.array(tour)
