r"""
Insertion Algorithm for TSP
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
from ml4co_kit.solver.lib.insertion.c_tsp_insertion import c_insertion


def tsp_insertion(task_data: TSPTask):
    # Preparation 
    points = task_data.points
    nodes_num = points.shape[0]
    _points = points.reshape(-1)
    
    # Random index
    index = np.arange(1, nodes_num)
    np.random.shuffle(index)
    random_index = np.insert(index, [0, len(index)], [0, 0])
    random_index = random_index.astype(np.int16)
    
    # Call ``c_insertion`` to get the tour
    insertion_tour = c_insertion(
        random_index.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),  
        _points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nodes_num
    )
    tour = np.ctypeslib.as_array(insertion_tour, shape=(nodes_num+1,))
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)