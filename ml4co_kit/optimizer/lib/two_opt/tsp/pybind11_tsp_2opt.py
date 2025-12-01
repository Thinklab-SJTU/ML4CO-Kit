"""
PyBind11-backed 2-opt local search for TSP.
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


import copy
import numpy as np
from typing import List
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.utils.type_utils import to_numpy
from .pybind11_impl import pybind11_tsp_2opt_impl


def _pybind11_tsp_2opt_ls(
    init_tours: np.ndarray,
    dists: np.ndarray,
    max_iters: int,
    type_2opt: int = 2,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Two-opt local search for TSP problems using the PyBind11 extension.

    Args:
        init_tour: (B, V+1) or (B, V+1)
        dists: (B, V, V) or (V, V)
        max_iters: Maximum number of iterations.
        type_2opt: Type of 2-opt local search.
    """
    # Prepare data
    tours = copy.deepcopy(init_tours.astype(np.int32))
    dists = dists.astype(np.float32)

    # Perform local search
    pybind11_tsp_2opt_impl(
        tours, dists, max_iters, type_2opt, num_workers
    )

    # Return the optimized tours
    return tours


def pybind11_tsp_2opt_ls(
    task_data: TSPTask,
    max_iters: int = 5000,
    type_2opt: int = 2
) -> None:
    """Run 2-opt local search and update the task data in-place."""
    # Get data from task data
    init_tour = to_numpy(task_data.sol)
    dists = to_numpy(task_data._get_dists())

    # Perform local search
    optimized_tour = _pybind11_tsp_2opt_ls(
        init_tours=init_tour,
        dists=dists,
        max_iters=max_iters,
        type_2opt=type_2opt
    )

    # Store the optimized tour in the task data
    task_data.from_data(sol=optimized_tour, ref=False)


def pybind11_tsp_2opt_batch_ls(
    batch_task_data: List[TSPTask],
    max_iters: int = 5000,
    type_2opt: int = 2
) -> None:
    """Run 2-opt local search and update the task data in-place."""
    # Check all task_data have the same number of nodes
    nodes_num = batch_task_data[0].nodes_num
    if not all(task_data.nodes_num == nodes_num for task_data in batch_task_data):
        raise ValueError("All task_data must have the same number of nodes.")
    
    # Get data from task data
    init_tours = np.array([task_data.sol for task_data in batch_task_data])
    dists = np.array([to_numpy(task_data._get_dists()) for task_data in batch_task_data])

    # Perform local search
    optimized_tours = _pybind11_tsp_2opt_ls(
        init_tours=init_tours,
        dists=dists,
        max_iters=max_iters,
        type_2opt=type_2opt,
        num_workers=len(batch_task_data)
    )
    
    # Store the optimized tour in the task data
    for task_data, tour in zip(batch_task_data, optimized_tours):
        task_data.from_data(sol=tour, ref=False)