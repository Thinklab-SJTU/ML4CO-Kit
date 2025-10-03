r"""
Global Prediction Degree Solver for MCL
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
from ml4co_kit.solver.lib.greedy.mcl_greedy import mcl_greedy


def mcl_gp_degree_decoder(task_data: MClTask):
    # Preparation for decoding
    adj_matrix = task_data.to_adj_matrix()
    adj_matrix_weighted = adj_matrix * task_data.nodes_weight
    weighted_degrees: np.ndarray = adj_matrix_weighted.sum(1)
    
    # The more connections, the more likely to be in the solution
    heatmap = weighted_degrees
    task_data.cache["heatmap"] = heatmap
    
    # Call Greedy Algorithm for MCl
    return mcl_greedy(task_data=task_data)