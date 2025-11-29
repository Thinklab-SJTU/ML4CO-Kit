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


def _mis_greedy(
    heatmap: np.ndarray, adj_matrix: np.ndarray
) -> np.ndarray:
    # Preparation
    np.fill_diagonal(adj_matrix, 0)
    sol = np.zeros_like(heatmap).astype(np.bool_)
    mask = np.zeros_like(heatmap).astype(np.bool_)
    sorted_nodes = np.argsort(-heatmap)
    
    # Greedy Algorithm for MIS
    for node in sorted_nodes:
        if not mask[node]:
            connect_nodes = np.where(adj_matrix[node] == 1)[0]
            sol[connect_nodes] = False
            sol[node] = True
            mask[connect_nodes] = True
            mask[node] = True
    sol = sol.astype(np.int32)
    return sol


def mis_greedy(task_data: MISTask):
    # Call ``_mis_greedy`` to get the solution
    sol = _mis_greedy(
        heatmap=task_data.cache["heatmap"],
        adj_matrix=task_data.to_adj_matrix()
    )
    
    # Store the solution in the task_data
    task_data.from_data(sol=sol, ref=False)