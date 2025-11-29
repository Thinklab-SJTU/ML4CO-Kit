r"""
Greedy Algorithm for MCl
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
from ml4co_kit.task.graph.mcl import MClTask


def _mcl_greedy(
    heatmap: np.ndarray, adj_matrix: np.ndarray
) -> np.ndarray:
    # Preparation
    np.fill_diagonal(adj_matrix, 1)
    sol = np.zeros_like(heatmap).astype(np.bool_)
    mask = np.zeros_like(heatmap).astype(np.bool_)
    sorted_nodes = np.argsort(-heatmap)
    
    # Greedy Algorithm for MCl
    for node in sorted_nodes:
        if not mask[node]:
            unconnect_nodes = np.where(adj_matrix[node] == 0)[0]
            sol[unconnect_nodes] = False
            sol[node] = True
            mask[unconnect_nodes] = True
            mask[node] = True
    sol = sol.astype(np.int32)
    return sol


def mcl_greedy(task_data: MClTask):
    # Call ``_mcl_greedy`` to get the solution
    sol = _mcl_greedy(
        heatmap=task_data.cache["heatmap"],
        adj_matrix=task_data.to_adj_matrix()
    )
    
    # Store the solution in the task_data
    task_data.from_data(sol=sol, ref=False)