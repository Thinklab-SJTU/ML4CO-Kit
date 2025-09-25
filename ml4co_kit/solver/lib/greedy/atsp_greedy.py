r"""
Greedy Algorithm for ATSP
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


import ctypes
import numpy as np
from ml4co_kit.task.routing.atsp import ATSPTask
from ml4co_kit.solver.lib.greedy.c_atsp_greedy import c_atsp_greedy_decoder


def atsp_greedy(task_data: ATSPTask):
    # Preparation for decoding
    heatmap: np.ndarray = -task_data.cache["heatmap"]
    nodes_num = heatmap.shape[-1]
    tour = (ctypes.c_int * nodes_num)(*(list(range(nodes_num))))
    cost = ctypes.c_double(0)
    heatmap = (ctypes.c_double *(nodes_num**2))(*heatmap.reshape(nodes_num*nodes_num).tolist())
    
    # Call c_atsp_greedy_decoder to get the tour
    c_atsp_greedy_decoder(nodes_num, heatmap, tour, ctypes.byref(cost))
    tour = np.array(list(tour))
    tour = np.append(tour, tour[0])

    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)