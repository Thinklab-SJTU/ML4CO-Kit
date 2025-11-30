r"""
Insertion Algorithm for MCut
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
from ml4co_kit.task.graph.mcut import MCutTask


def mcut_insertion(task_data: MCutTask):
    # Preparation
    adj_matrix_weighted = task_data.to_adj_matrix(with_edge_weights=True)
    nodes_num = adj_matrix_weighted.shape[0]
    np.fill_diagonal(adj_matrix_weighted, 0)  # Remove self-loops
    
    # Random index - generate a random sequence of nodes
    index = np.arange(nodes_num)
    np.random.shuffle(index)
    
    # Initialize solution: set_A is represented by sol=1, set_B by sol=0
    sol = np.zeros(nodes_num, dtype=np.bool_)
    
    # Put the first node in set A
    sol[index[0]] = True
    
    # Insertion: for each remaining node, decide which set to put it in
    # by checking which assignment gives more cut edges
    for i in range(1, nodes_num):
        node = index[i]
        
        # Calculate degree to set A and set B
        degree_to_A = adj_matrix_weighted[node][sol].sum()
        degree_to_B = adj_matrix_weighted[node][~sol].sum()
        
        # Assign to the set that maximizes the cut
        # If the node has more connections to A, put it in B (sol=0)
        # If the node has more connections to B, put it in A (sol=1)
        if degree_to_B > degree_to_A:
            sol[node] = True  # Put in set A
        else:
            sol[node] = False  # Put in set B
    
    # Store the solution in the task_data
    sol = sol.astype(np.int32)
    task_data.from_data(sol=sol, ref=False)

