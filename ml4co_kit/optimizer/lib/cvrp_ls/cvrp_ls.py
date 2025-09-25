r"""
CVRP local search algorithm.
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
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.optimizer.lib.cvrp_ls.c_classic import c_cvrp_local_search


def cvrp_ls(
    task_data: CVRPTask,
    coords_scale: int = 1000,
    demands_scale: int = 1000,
    seed: int = 1234
):
    """Classic local search for CVRP problems using C implementation."""
    # Preparation
    init_tour = task_data.sol.astype(np.int16)
    coords = task_data.coords.astype(np.float32).reshape(-1)
    norm_demands = task_data.norm_demands.astype(np.float32).reshape(-1)
    nodes_num = coords.shape[0]
    
    # Perform local search
    ls_tour = c_cvrp_local_search(
        init_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
        norm_demands.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        nodes_num,
        len(init_tour),
        coords_scale,
        demands_scale,
        seed
    )
    ls_tour = np.ctypeslib.as_array(ls_tour, shape=(len(init_tour)+2,))
    
    # Store the optimized tour in the task data
    if ls_tour[0] != -1:
        ls_tour = ls_tour[:np.where(ls_tour==-1)[0][0]]
        task_data.from_data(sol=ls_tour, ref=False)