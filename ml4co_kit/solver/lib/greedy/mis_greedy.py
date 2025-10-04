r"""
Greedy Algorithm for MIS
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


def mis_greedy(task_data: MISTask):
    # Preparation for decoding
    heatmap: np.ndarray = task_data.cache["heatmap"]
    adj_matrix = task_data.to_adj_matrix()
    sol = np.zeros_like(heatmap).astype(np.bool_)
    mask = np.zeros_like(heatmap).astype(np.bool_)
    sorted_nodes = np.argsort(-heatmap)
    
    # Greedy Algorithm for MIS
    for node in sorted_nodes:
        if not mask[node]:
            if (adj_matrix[node][sol]).sum() == 0:
                connect_nodes = np.where(adj_matrix[node] == 1)[0]
                sol[connect_nodes] = False
                sol[node] = True
                mask[connect_nodes] = True
                mask[node] = True
    sol = sol.astype(np.int32)
    
    # Store the solution in the task_data
    task_data.from_data(sol=sol, ref=False)