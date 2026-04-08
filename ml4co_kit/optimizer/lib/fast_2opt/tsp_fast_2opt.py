"""
PyBind11-backed fast 2-opt local search for TSP.
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


import numpy as np
from ml4co_kit.task.routing.tsp import TSPTask
from .pybind11_tsp_fast_2opt import c_tsp_fast_2opt_impl


def _pybind11_tsp_fast_2opt_ls(
    tour: np.ndarray,
    points: np.ndarray,
    num_steps: int = -1,
    knn: int = 50,
    num_workers: int = 1,
    seed: int = 1234,
) -> np.ndarray:
    """Optimize tour in-place.
    """
    return c_tsp_fast_2opt_impl(
        tour, points, int(num_steps), knn, num_workers, int(seed)
    )


def pybind11_tsp_fast_2opt_ls(
    task_data: TSPTask,
    num_steps: int = -1,
    knn: int = 50,
    num_workers: int = 1,
    seed: int = 1234,
) -> None:
    """Run fast 2-opt"""
    # Get data from task data
    tour = task_data.sol
    points = task_data.points.astype(np.float32)
    
    # Call ``_pybind11_tsp_fast_2opt_ls`` to optimize the tour
    optimized_tour = _pybind11_tsp_fast_2opt_ls(
        tour=tour,
        points=points,
        num_steps=num_steps,
        knn=knn,
        num_workers=num_workers,
        seed=seed,
    )

    # Store the optimized tour in the task data
    task_data.from_data(sol=optimized_tour, ref=False)
