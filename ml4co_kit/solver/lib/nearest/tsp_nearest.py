r"""
Nearest Neighbor Algorithm for TSP
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
from ml4co_kit.task.routing.tsp import TSPTask
from .c_tsp_nearest import pybind11_tsp_nearest_impl

def tsp_nearest(task_data: TSPTask):
    # Preparation
    points = task_data.points.astype(np.float32)

    # Call PyBind11 nearest-neighbor implementation
    tour = pybind11_tsp_nearest_impl(points)
    tour = np.asarray(tour, dtype=np.int32)

    # Store the tour in the task data
    task_data.from_data(sol=tour, ref=False)