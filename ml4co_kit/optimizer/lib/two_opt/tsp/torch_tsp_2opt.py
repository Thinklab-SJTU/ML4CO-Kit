r"""
Two-Opt local search algorithm for TSP.
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


import torch
import numpy as np
from typing import List
from torch import Tensor
from ml4co_kit.task.routing.base import DISTANCE_TYPE
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.utils.type_utils import to_numpy, to_tensor


def _torch_tsp_2opt_ls(
    init_tours: Tensor,
    points: Tensor,
    max_iters: int,
    device: str = "cpu"
):
    """
    Two-Opt local search for TSP problems.

    Args:
        init_tours: (B, V+1)
        points: (V, 2) or (B, V, 2)
        max_iters: Maximum number of iterations.
        device: Device to run on.
    """
    # Init iterator
    iterator = 0 

    # Preparation
    tours = init_tours.to(device)
    points = points.to(device)
    
    # Get batch size and number of nodes
    batch_size = tours.shape[0]
    num_nodes = tours.shape[1] - 1  # V+1 -> V
    
    # Handle points shape: support both (V, 2) and (B, V, 2)
    if points.dim() == 2:
        # (V, 2) -> expand to (B, V, 2) for batch processing
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
    elif points.dim() == 3:
        # (B, V, 2) -> already in correct shape
        if points.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: tours has {batch_size} batches, but points has {points.shape[0]} batches")
        if points.shape[1] != num_nodes:
            raise ValueError(f"Number of nodes mismatch: tours has {num_nodes} nodes, but points has {points.shape[1]} nodes")
    else:
        raise ValueError(f"Invalid points shape: {points.shape}, expected (V, 2) or (B, V, 2)")
    
    # Start 2opt
    with torch.inference_mode():
        while True:
            # Get points for each position in tours
            # tours: (B, V+1), points: (B, V, 2)
            # We need to index points using tours indices
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_nodes)
            tour_points = points[batch_indices, tours[:, :-1]]  # (B, V, 2) - points at each tour position
            tour_points_next = points[batch_indices, tours[:, 1:]]  # (B, V, 2) - next points in tour
            
            # Expand dimensions for pairwise distance computation
            # For 2-opt, we compare edge (i, i+1) with edge (j, j+1)
            # points_i: (B, V, 2) -> (B, V, 1, 2)
            # points_j: (B, V, 2) -> (B, 1, V, 2)
            points_i = tour_points.unsqueeze(2)  # (B, V, 1, 2)
            points_j = tour_points.unsqueeze(1)  # (B, 1, V, 2)
            points_i_plus_1 = tour_points_next.unsqueeze(2)  # (B, V, 1, 2)
            points_j_plus_1 = tour_points_next.unsqueeze(1)  # (B, 1, V, 2)
            
            # Distance matrix
            # A_ij: distance from point i to point j (after swap)
            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, dim=-1))  # (B, V, V)
            # A_i_plus_1_j_plus_1: distance from point i+1 to point j+1 (after swap)
            A_i_plus_1_j_plus_1 = torch.sqrt(
                torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, dim=-1)
            )  # (B, V, V)
            # A_i_i_plus_1: current distance from point i to point i+1
            A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, dim=-1))  # (B, V, V)
            # A_j_j_plus_1: current distance from point j to point j+1
            A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, dim=-1))  # (B, V, V)
            
            # Change
            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1  # (B, V, V)
            valid_change = torch.triu(change, diagonal=2)  # (B, V, V)

            # Min change for each batch
            # Find the minimum change across all batches
            valid_change_flat = valid_change.reshape(batch_size, -1)  # (B, V*V)
            min_change_per_batch = torch.min(valid_change_flat, dim=-1)[0]  # (B,)
            min_change = torch.min(min_change_per_batch)  # scalar
            
            # Check if any batch has improvement
            if min_change < -1e-6:
                # Find argmin for each batch
                flatten_argmin_index = torch.argmin(valid_change_flat, dim=-1)  # (B,)
                min_i = torch.div(flatten_argmin_index, num_nodes, rounding_mode='floor')  # (B,)
                min_j = torch.remainder(flatten_argmin_index, num_nodes)  # (B,)
                
                # Apply 2-opt swap for each batch that has improvement
                for b in range(batch_size):
                    if min_change_per_batch[b] < -1e-6:
                        i_idx = min_i[b].item()
                        j_idx = min_j[b].item()
                        if i_idx < j_idx:
                            tours[b, i_idx + 1:j_idx + 1] = torch.flip(
                                tours[b, i_idx + 1:j_idx + 1], dims=(0,)
                            )
                iterator += 1
            else:
                break
                
            # Check iteration
            if iterator >= max_iters:
                break
    
    # Return the optimized tour
    return tours


def torch_tsp_2opt_ls(
    task_data: TSPTask, 
    max_iters: int = 5000, 
    device: str = "cpu"
):
    """Two-Opt local search for TSP problems."""
    # Get data from task data
    init_tours = task_data.sol
    points = task_data.points
    
    # Preparation
    init_tours = np.expand_dims(init_tours, axis=0)
    tours: Tensor = to_tensor(init_tours).to(device)
    points: Tensor = to_tensor(points).to(device)

    # Perform local search
    tours = _torch_tsp_2opt_ls(
        init_tours=tours,
        points=points,
        max_iters=max_iters,
        device=device
    )
    
    # Store the optimized tour in the task data
    optimized_tour = to_numpy(tours[0])
    task_data.from_data(sol=optimized_tour, ref=False)


def torch_tsp_2opt_batch_ls(
    batch_task_data: List[TSPTask], 
    max_iters: int = 5000, 
    device: str = "cpu"
):
    """Two-Opt local search for TSP problems."""
    # Check the distance type is EUC_2D and the number of nodes is the same
    nodes_num = batch_task_data[0].nodes_num
    for task_data in batch_task_data:
        if task_data.dist_eval.distance_type != DISTANCE_TYPE.EUC_2D:
            raise ValueError("Distance type must be EUC_2D.")
        if task_data.nodes_num != nodes_num:
            raise ValueError("All task_data must have the same number of nodes.")

    # Get data from task data
    tours = np.array([task_data.sol for task_data in batch_task_data])
    points = np.array([task_data.points for task_data in batch_task_data])
    tours: Tensor = to_tensor(tours).to(device)
    points: Tensor = to_tensor(points).to(device)

    # Perform local search
    optimized_tours = _torch_tsp_2opt_ls(
        init_tours=tours,
        points=points,
        max_iters=max_iters,
        device=device
    )
    
    # Store the optimized tour in the task data
    optimized_tours = to_numpy(optimized_tours)
    for task_data, tour in zip(batch_task_data, optimized_tours):
        task_data.from_data(sol=tour, ref=False)