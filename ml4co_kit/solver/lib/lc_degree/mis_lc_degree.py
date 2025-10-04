r"""
Local Construction Degree Solver for MIS
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


import copy
import numpy as np
from ml4co_kit.task.graph.mis import MISTask


def mis_lc_degree(task_data: MISTask):
    # Preparation for decoding
    adj = task_data.to_adj_matrix()
    adj_matrix = copy.deepcopy(adj)
    np.fill_diagonal(adj_matrix, 0) # Remove self-loops
    lc_graph = adj_matrix * task_data.nodes_weight
    degrees: np.ndarray = lc_graph.sum(1) - task_data.nodes_weight
    sol = np.zeros_like(degrees).astype(np.bool_)
    mask = np.zeros_like(degrees).astype(np.bool_)
    
    # Each step, find the node with the minimum degree
    # Until all nodes are masked
    while not mask.all():
        next_node = np.argmin(degrees)
        connect_nodes = np.where(adj_matrix[next_node] == 1)[0]
        sol[connect_nodes] = False
        sol[next_node] = True
        mask[connect_nodes] = True
        mask[next_node] = True
        adj_matrix[connect_nodes, :] = 0
        adj_matrix[:, connect_nodes] = 0
        adj_matrix[next_node, :] = 0
        adj_matrix[:, next_node] = 0
        lc_graph = adj_matrix * task_data.nodes_weight
        degrees = lc_graph.sum(1) - task_data.nodes_weight
        degrees[mask] = 1000000.0
    
    # Store the solution in the task_data
    sol = sol.astype(np.int32)
    task_data.from_data(sol=sol, ref=False)