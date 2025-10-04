r"""
Greedy Algorithm for CVRP
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from ml4co_kit.task.routing.cvrp import CVRPTask


def cvrp_greedy(task_data: CVRPTask):
    # Preparation for decoding
    heatmap: np.ndarray = task_data.cache["heatmap"]
    np.fill_diagonal(heatmap, 0)
    norm_demand = task_data.norm_demands
    nodes_visited = np.zeros_like(norm_demand).astype(np.bool_)
    tour = [0]
    current_node = 0
    nodes_visited[0] = True
    
    # Greedy Algorithm for CVRP
    while not nodes_visited.all():
        # find next node
        if current_capacity >= 0.5:
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
    tour.append(0)
    tour = np.array(tour)
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)