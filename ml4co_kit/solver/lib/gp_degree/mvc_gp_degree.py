r"""
Global Prediction Degree Solver for MVC
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
from ml4co_kit.task.graph.mvc import MVCTask
from ml4co_kit.solver.lib.greedy.mvc_greedy import mvc_greedy


def mvc_gp_degree_decoder(task_data: MVCTask):
    # Preparation for decoding
    adj_matrix = task_data.to_adj_matrix()
    adj_matrix_weighted = adj_matrix * task_data.nodes_weight
    weighted_degrees: np.ndarray = adj_matrix_weighted.sum(1)
    
    # The more connections, the more likely to be in the solution
    heatmap = weighted_degrees - task_data.nodes_weight
    task_data.cache["heatmap"] = heatmap

    # Call Greedy Algorithm for MVC
    return mvc_greedy(task_data=task_data)