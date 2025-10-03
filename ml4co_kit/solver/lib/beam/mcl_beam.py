r"""
Beam Algorithm for MCl
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


def mcl_beam(task_data: MClTask, beam_size: int = 16):
    # Preparation for decoding
    empty_flag = [True for _ in range(beam_size)]
    clique = [list() for _ in range(beam_size)] 
    heatmap: np.ndarray = task_data.cache["heatmap"]
    adj_matrix = task_data.to_adj_matrix()
    np.fill_diagonal(adj_matrix, 1)
    sol = np.zeros_like(heatmap)
    beam_sols = np.repeat(sol.reshape(1, -1), beam_size, axis=0)
    beam_sols_weighted = np.repeat(sol.reshape(1, -1), beam_size, axis=0)
    mask = np.zeros_like(heatmap).astype(np.bool_)
    sorted_nodes = np.argsort(-heatmap)
    nodes_weight = task_data.nodes_weight
    
    # Greedy Algorithm for MCl
    for node in sorted_nodes:
        if not mask[node]:
            for idx in range(beam_size):
                if empty_flag[idx]:
                    clique[idx].append(node)
                    beam_sols[idx][node] = 1
                    beam_sols_weighted[idx][node] = nodes_weight[node]
                    empty_flag[idx] = False
                    break
                if (adj_matrix[node][clique[idx]] == 1).all():
                    clique[idx].append(node)
                    beam_sols[idx][node] = 1
                    beam_sols_weighted[idx][node] = nodes_weight[node]
                    break
                
    best_idx = np.argmax(beam_sols_weighted.sum(axis=1))
    sol: np.ndarray = beam_sols[best_idx]
    sol = sol.astype(np.int32)
    
    # Store the solution in the task_data
    task_data.from_data(sol=sol, ref=False)