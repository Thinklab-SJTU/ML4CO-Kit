r"""
MCTS Algorithm for TSP
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
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE
from ml4co_kit.solver.lib.mcts.c_tsp_mcts import c_mcts_decoder


def _tsp_mcts(
    points: np.ndarray,
    heatmap: np.ndarray,
    mcts_time_limit: float = 1.0,
    mcts_max_depth: int = 10, 
    mcts_type_2opt: int = 1, 
    mcts_max_iterations_2opt: int = 5000
) -> np.ndarray:
    # Preparation
    nodes_num = heatmap.shape[-1]
    _heatmap = heatmap.astype(np.float32).reshape(-1)
    _points = points.astype(np.float32).reshape(-1)
    
    # Call ``c_mcts_decoder`` to get the tour
    tour = c_mcts_decoder(
        _heatmap.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
        _points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        nodes_num,
        mcts_max_depth,
        ctypes.c_float(mcts_time_limit),
        mcts_type_2opt,
        mcts_max_iterations_2opt,
    )
    
    # Convert the tour to numpy array
    tour = np.ctypeslib.as_array(tour, shape=(nodes_num,))
    tour = np.append(tour, 0)
    return tour


def tsp_mcts(
    task_data: TSPTask,
    mcts_time_limit: float = 1.0,
    mcts_max_depth: int = 10, 
    mcts_type_2opt: int = 1, 
    mcts_max_iterations_2opt: int = 5000
):
    # Check the distance type
    if task_data.distance_type != DISTANCE_TYPE.EUC_2D:
        raise NotImplementedError(
            f"MCTS is only supported for EUC_2D distance type, "
            f"but got {task_data.distance_type} instead."
        )

    # Call ``_tsp_mcts`` to get the tour
    tour = _tsp_mcts(
        points=task_data.points,
        heatmap=task_data.cache["heatmap"],
        mcts_time_limit=mcts_time_limit,
        mcts_max_depth=mcts_max_depth,
        mcts_type_2opt=mcts_type_2opt,
        mcts_max_iterations_2opt=mcts_max_iterations_2opt,
    )
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)