r"""
Greedy Algorithm for TSP
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
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.solver.lib.greedy.cython_tsp_greedy import cython_tsp_greedy


def tsp_greedy(task_data: TSPTask):
    # Preparation for decoding
    heatmap: np.ndarray = task_data.cache["heatmap"]
    heatmap = heatmap.astype("double")
    
    # Call cython_tsp_greedy to get the adjacency matrix
    adj_mat = cython_tsp_greedy(heatmap)[0]
    adj_mat = np.asarray(adj_mat)
    
    # Get the tour from the adjacency matrix
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
    tour = np.array(tour)
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)