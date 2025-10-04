r"""
Local Construction Degree Solver for MCut
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
from ml4co_kit.task.graph.mcut import MCutTask


def mcut_lc_degree(task_data: MCutTask):
    # Preparation for decoding
    adj_matrix_weighted = task_data.to_adj_matrix(with_edge_weights=True)
    lc_graph = copy.deepcopy(adj_matrix_weighted)
    np.fill_diagonal(lc_graph, 0) # Remove self-loops
    nodes_num = lc_graph.shape[0]
    mask = np.zeros(shape=(nodes_num,)).astype(np.bool_)
    set_A = np.zeros(shape=(nodes_num,)).astype(np.bool_)
    set_B = np.zeros(shape=(nodes_num,)).astype(np.bool_)    
    set_A[0] = True # default
    mask[0] = True # default
    
    # Each step, find the node that can maximize the 
    # difference of degrees between set A and set B
    while not mask.all():
        # get degree
        degree_A = lc_graph[set_A].sum(0)
        degree_B = lc_graph[set_B].sum(0)
        degree_A[mask] = -1
        degree_B[mask] = -1
        
        # select next node and update
        max_A = np.max(degree_A)
        max_B = np.max(degree_B)
        if max_A > max_B:
            next_node = np.argmax(degree_A)
            set_B[next_node] = True
            mask[next_node] = True
        else:
            next_node = np.argmax(degree_B)
            set_A[next_node] = True
            mask[next_node] = True
            
    # Store the solution in the task_data
    sol = set_A.astype(np.int32)
    task_data.from_data(sol=sol, ref=False)