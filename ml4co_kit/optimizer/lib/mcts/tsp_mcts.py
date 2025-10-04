r"""
MCTS local search algorithm for TSP.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import ctypes
import numpy as np
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.optimizer.lib.mcts.c_tsp_mcts import c_mcts_local_search


def tsp_mcts_ls(
    task_data: TSPTask,
    mcts_time_limit: float = 1.0,
    mcts_max_depth: int = 10, 
    mcts_type_2opt: int = 1,
    mcts_continue_flag: int = 2,
    mcts_max_iterations_2opt: int = 5000
):
    # Preparation
    heatmap: np.ndarray = task_data.cache["heatmap"]
    nodes_num = heatmap.shape[-1]
    init_tour = task_data.sol.astype(np.int16)
    heatmap = heatmap.astype(np.float32).reshape(-1)
    points = task_data.points.astype(np.float32).reshape(-1)
    points = points.reshape(-1)
    heatmap = heatmap.reshape(-1)  
    
    # Call ``c_mcts_local_search`` to optimize the tour
    mcts_tour = c_mcts_local_search(
        init_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
        heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
        points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        nodes_num,
        mcts_max_depth,
        ctypes.c_float(mcts_time_limit),
        mcts_type_2opt,
        mcts_continue_flag,
        mcts_max_iterations_2opt,
    )
    mcts_tour = np.ctypeslib.as_array(mcts_tour, shape=(nodes_num,))
    mcts_tour = np.append(mcts_tour, 0)

    # Store the optimized tour in the task data
    task_data.from_data(sol=mcts_tour, ref=False)