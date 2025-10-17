r"""
Two-opt local search algorithm for TSP.
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
from ml4co_kit.task.routing.atsp import ATSPTask
from ml4co_kit.optimizer.lib.two_opt.c_atsp_2opt import c_atsp_2opt_local_search


def atsp_2opt_ls(
    task_data: ATSPTask, 
    max_iters: int = 5000,
):
    """Two-opt local search for ATSP problems."""
    # Get data from task data
    init_tour = task_data.sol.astype(np.int16)
    dists = task_data.dists.astype(np.float32)
    nodes_num = dists.shape[-1]
    
    # Perform local search
    ls_tour = c_atsp_2opt_local_search(
        init_tour.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
        dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        nodes_num,
        max_iters,
    )
    ls_tour = np.ctypeslib.as_array(ls_tour, shape=(nodes_num,))
    ls_tour = np.append(ls_tour, 0)
    
    # Store the optimized tour in the task data
    task_data.from_data(sol=ls_tour, ref=False)