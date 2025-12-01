r"""
Insertion Algorithm for MIS
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
from ml4co_kit.task.graph.mis import MISTask


def mis_insertion(task_data: MISTask):
    # Preparation
    adj_matrix = task_data.to_adj_matrix()
    nodes_num = adj_matrix.shape[0]
    np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
    
    # Random index - generate a random sequence of nodes
    index = np.arange(nodes_num)
    np.random.shuffle(index)
    
    # Initialize solution
    sol = np.zeros(nodes_num, dtype=np.bool_)
    mask = np.zeros(nodes_num, dtype=np.bool_)

    # Greedy Algorithm for MIS
    for node in index:
        if not mask[node]:
            connect_nodes = np.where(adj_matrix[node] == 1)[0]
            sol[connect_nodes] = False
            sol[node] = True
            mask[connect_nodes] = True
            mask[node] = True
    sol = sol.astype(np.int32)
    
    # Store the solution in the task_data
    task_data.from_data(sol=sol, ref=False)