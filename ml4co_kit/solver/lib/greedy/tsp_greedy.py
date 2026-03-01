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
from ml4co_kit.solver.lib.greedy.pybind11_tsp_greedy_v2 import pybind11_tsp_greedy_v2


def _tsp_greedy(heatmap: np.ndarray) -> np.ndarray:
    # Call cython_tsp_greedy to get the adjacency matrix
    heatmap = heatmap.astype("double")
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
    return tour


def tsp_greedy(task_data: TSPTask):
    # Call ``_tsp_greedy`` to get the tour
    tour = _tsp_greedy(heatmap=task_data.cache["heatmap"])
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)


def _tsp_greedy_v2(
    candidate_edges: np.ndarray, nodes_num: int, num_workers: int = 1
) -> np.ndarray:
    # Ensure input is int32 for C++ compatibility
    candidate_edges = np.ascontiguousarray(candidate_edges, dtype=np.int32)
    
    # Call the pybind11 implementation
    tours = pybind11_tsp_greedy_v2(candidate_edges, nodes_num, num_workers)
    
    return tours


def tsp_greedy_v2(task_data: TSPTask):
    # Call ``_tsp_greedy_v2`` to get the tour
    tour = _tsp_greedy_v2(
        candidate_edges=task_data.cache["candidate_edges"], 
        nodes_num=task_data.nodes_num, 
        num_workers=1
    )
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)