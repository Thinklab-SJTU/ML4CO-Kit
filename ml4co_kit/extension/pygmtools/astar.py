r"""
A* Algorithm for Graph Matching.

This module implements the A* search algorithm for solving the Quadratic Assignment
Problem (QAP) in graph matching. A* is a best-first search algorithm that uses a
heuristic function to guide the search towards the optimal solution.

The algorithm explores the solution space using a priority queue, where nodes are
prioritized based on their cost (g) plus a heuristic estimate (h) of the remaining
cost. This implementation uses the Hungarian algorithm as the heuristic function.

Reference:
    Wang, R., Yan, J., & Yang, X. (2021). Learning combinatorial embedding networks
    for deep graph matching. In ICCV 2021.
"""

# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import torch
import functools
from torch import Tensor
from typing import Dict, Optional
from .c_astar import cpp_astar
from .hungarian import pygm_hungarian
from .utils import _check_and_init_gm


def hungarian_ged(node_cost_mat: Tensor, n1: int, n2: int) -> tuple:
    """
    Compute Graph Edit Distance (GED) lower bound using Hungarian algorithm.
    
    This function solves a modified assignment problem to compute the lower bound
    of the Graph Edit Distance between two graphs. It constructs an augmented cost
    matrix that allows for node deletions and insertions, then applies the Hungarian
    algorithm to find the optimal assignment.
    
    Args:
        node_cost_mat: Node cost matrix of shape (n1+1, n2+1)
                       node_cost_mat[i, j] is the cost of matching node i to node j
                       The last row/column represents the cost of leaving nodes unmatched
        n1: Number of nodes in graph 1 (excluding dummy node)
        n2: Number of nodes in graph 2 (excluding dummy node)
    
    Returns:
        Tuple containing:
        - pred_x: Predicted assignment matrix of shape (n1+1, n2+1)
                  Binary matrix where pred_x[i,j]=1 means node i matches node j
        - ged_lower_bound: Lower bound of the Graph Edit Distance (scalar)
    
    Raises:
        RuntimeError: If node_cost_mat dimensions don't match (n1+1, n2+1)
    
    Note:
        - The augmented matrix structure allows for partial matching
        - Diagonal elements in upper_right/lower_left represent deletion/insertion costs
        - This provides an admissible heuristic for A* search
    """
    # Validate input dimensions
    if not node_cost_mat.shape[-2] == n1 + 1:
        raise RuntimeError(
            f'node_cost_mat dimension mismatch in hungarian_ged. '
            f'Got {node_cost_mat.shape[-2]} in dim -2 but {n1 + 1} is expected'
        )
    if not node_cost_mat.shape[-1] == n2 + 1:
        raise RuntimeError(
            f'node_cost_mat dimension mismatch in hungarian_ged. '
            f'Got {node_cost_mat.shape[-1]} in dim -1 but {n2 + 1} is expected'
        )
    
    device = node_cost_mat.device
    
    # Construct augmented cost matrix for GED computation
    # Structure:
    #   [upper_left  | upper_right]
    #   [lower_left  | lower_right]
    
    # Upper left: direct matching costs between nodes
    upper_left = node_cost_mat[:n1, :n2]
    
    # Upper right: cost of deleting nodes from graph 1
    upper_right = torch.full((n1, n1), float('inf'), device=device)
    torch.diagonal(upper_right)[:] = node_cost_mat[:-1, -1]
    
    # Lower left: cost of inserting nodes into graph 1 (from graph 2)
    lower_left = torch.full((n2, n2), float('inf'), device=device)
    torch.diagonal(lower_left)[:] = node_cost_mat[-1, :-1]
    
    # Lower right: dummy nodes (zero cost)
    lower_right = torch.zeros((n2, n1), device=device)
    
    # Concatenate to form the augmented cost matrix
    large_cost_mat = torch.cat(
        (
            torch.cat((upper_left, upper_right), dim=1),
            torch.cat((lower_left, lower_right), dim=1)
        ), 
        dim=0
    )
    
    # Apply Hungarian algorithm (negate because hungarian maximizes)
    large_pred_x = pygm_hungarian(-large_cost_mat.unsqueeze(dim=0)).squeeze()
    
    # Extract the assignment matrix in original format
    pred_x = torch.zeros_like(node_cost_mat)
    pred_x[:n1, :n2] = large_pred_x[:n1, :n2]  # Direct matches
    pred_x[:-1, -1] = torch.sum(large_pred_x[:n1, n2:], dim=1)  # Deletions
    pred_x[-1, :-1] = torch.sum(large_pred_x[n1:, :n2], dim=0)  # Insertions
    
    # Compute GED lower bound as total cost of the assignment
    ged_lower_bound = torch.sum(pred_x * node_cost_mat)
    
    return pred_x, ged_lower_bound


def heuristic_prediction_hun(
    k: Tensor, 
    n1: int, 
    n2: int, 
    partial_pmat: Tensor, 
    cache_dict: Optional[Dict] = None
) -> Tensor:
    """
    Heuristic function for A* search using Hungarian algorithm.
    
    This function estimates the remaining cost for completing a partial matching
    by solving a reduced assignment problem on the unmatched nodes. It uses the
    Hungarian algorithm to compute the optimal cost for the remaining nodes.
    
    Args:
        k: Cost matrix for node pairs, shape varies based on problem size
           Represents the cost of matching remaining node pairs
        n1: Number of nodes in graph 1
        n2: Number of nodes in graph 2
        partial_pmat: Partial permutation matrix representing current matching
                      Shape: (n1+1, n2+1), where +1 accounts for dummy node
        cache_dict: Optional dictionary for caching node costs across calls
                    Helps avoid redundant computations
    
    Returns:
        Estimated remaining cost (Graph Edit Distance) as a scalar tensor
    
    Note:
        - The heuristic is admissible (never overestimates the true cost)
        - Caching significantly improves performance for repeated calls
        - The dummy dimension allows for partial matchings
    """
    # Check if node costs are cached
    if cache_dict is not None and 'node_cost' in cache_dict:
        node_cost_mat = cache_dict['node_cost']
    else:
        # Compute node costs using Hungarian algorithm
        k_prime = k.reshape(-1, n1 + 1, n2 + 1)
        node_costs = torch.empty(k_prime.shape[0])
        
        # Compute cost for each node pair
        for i in range(k_prime.shape[0]):
            _, node_costs[i] = hungarian_ged(k_prime[i], n1, n2)
        
        # Reshape to matrix form
        node_cost_mat = node_costs.reshape(n1 + 1, n2 + 1)
        
        # Cache the result for future use
        if cache_dict is not None:
            cache_dict['node_cost'] = node_cost_mat
    
    # Create masks for unmatched nodes
    # graph_1_mask[i] = True if node i in graph 1 is not yet matched
    graph_1_mask = ~partial_pmat.sum(dim=-1).to(dtype=torch.bool)
    graph_2_mask = ~partial_pmat.sum(dim=-2).to(dtype=torch.bool)
    
    # Always keep dummy node available
    graph_1_mask[-1] = 1
    graph_2_mask[-1] = 1
    
    # Extract costs only for unmatched nodes
    node_cost_mat = node_cost_mat[graph_1_mask, :]
    node_cost_mat = node_cost_mat[:, graph_2_mask]
    
    # Solve reduced assignment problem for remaining nodes
    _, ged = hungarian_ged(
        node_cost_mat, 
        torch.sum(graph_1_mask[:-1]),  # Number of unmatched nodes in graph 1
        torch.sum(graph_2_mask[:-1])   # Number of unmatched nodes in graph 2
    )
    
    return ged


def classic_astar_kernel(K_padded: Tensor, n1: int, n2: int, beam_width: int) -> Tensor:
    """
    Core A* search kernel implementation.
    
    This function performs the actual A* search using the provided affinity matrix
    and heuristic function. It explores the search space using beam search with
    the specified beam width.
    
    Args:
        K_padded: Padded affinity matrix with dummy dimensions
                  Shape: (padded_n1n2, padded_n1n2)
        n1: Number of nodes in graph 1
        n2: Number of nodes in graph 2
        beam_width: Number of best candidates to keep at each search level
                    Larger values explore more thoroughly but are slower
    
    Returns:
        Predicted assignment matrix of shape (n1+1, n2+1)
        The +1 dimension is for the dummy node (partial matching)
    
    Note:
        - This converts the maximization problem to minimization by negating K
        - Uses beam search to balance between optimality and efficiency
        - The trust_fact=1.0 means fully trust the heuristic function
    """

    # Create cache dictionary for heuristic function
    cache_dict = {}
    hun_func = functools.partial(heuristic_prediction_hun, cache_dict=cache_dict)
    
    # Run A* search
    x_pred, _ = cpp_astar(
        None,
        -K_padded,  # Negate to convert maximization to minimization
        n1, n2,
        None,
        hun_func,
        net_pred=False,      # Don't use neural network prediction
        beam_width=beam_width,
        trust_fact=1.,       # Fully trust the heuristic
        no_pred_size=0,      # Always use prediction
    )
    
    return x_pred


def pygm_astar(
    K: Tensor, 
    n1: Tensor, 
    n2: Tensor, 
    n1max: int, 
    n2max: int, 
    beam_width: int
) -> Tensor:
    """
    PyTorch implementation of A* algorithm for solving QAP in graph matching.
    
    A* is an informed search algorithm that finds the optimal assignment by
    exploring the solution space guided by a heuristic function. This implementation
    uses the Hungarian algorithm as the heuristic to estimate the remaining cost.
    
    The algorithm guarantees finding the optimal solution when the heuristic is
    admissible (never overestimates). The beam width parameter controls the
    trade-off between solution quality and computational cost.
    
    Args:
        K: Affinity matrix of shape (batch_size, n1max*n2max, n1max*n2max)
           K[b, i*n2max+j, k*n2max+l] represents the affinity between 
           matching (i,j) and matching (k,l) in batch b
        n1: Number of nodes in graph 1 for each batch, shape (batch_size,)
            If None, uses n1max for all batches
        n2: Number of nodes in graph 2 for each batch, shape (batch_size,)
            If None, uses n2max for all batches
        n1max: Maximum number of nodes in graph 1 across all batches
        n2max: Maximum number of nodes in graph 2 across all batches
        beam_width: Number of best candidates to keep at each search level
                    Typical values: 1 (greedy), 10-100 (balanced), 1000+ (thorough)
    
    Returns:
        Predicted assignment matrix of shape (batch_size, n1max, n2max)
        Values are binary (0 or 1) representing the optimal matching
    
    Raises:
        ValueError: If any batch has n1 > n2 (algorithm requires n1 <= n2)
    
    Note:
        - Requires n1 <= n2 for all batches (algorithm constraint)
        - Larger beam_width gives better solutions but takes longer
        - beam_width=1 is equivalent to greedy search
        - The algorithm uses column-wise vectorization internally
        - Dummy dimensions are added to support partial matchings
    """
    # Initialize and validate parameters
    batch_num, n1, n2, n1max, n2max, n1n2, _ = _check_and_init_gm(K, n1, n2, n1max, n2max, None)
    
    # Validate constraint: n1 <= n2
    if torch.any(n1 > n2):
        raise ValueError(
            'Number of nodes in graph 1 should always be <= number of nodes in graph 2. '
            'Consider swapping the graphs if n1 > n2.'
        )
    
    # Initialize output tensor
    x_pred = torch.zeros((batch_num, n1max, n2max), device=K.device)
    
    # Reshape K from vectorized form to 4D tensor
    # Original: (batch, n1max*n2max, n1max*n2max)
    # Reshaped: (batch, n2max, n1max, n2max, n1max)
    # Note: This accounts for row-wise vs column-wise vectorization difference
    K = K.reshape(batch_num, n2max, n1max, n2max, n1max)
    
    # Process each batch separately
    for b in range(batch_num):
        # Extract the real affinity matrix (excluding padding)
        real_K = K[b, :n2[b], :n1[b], :n2[b], :n1[b]]
        
        # Add dummy dimensions for partial matching support
        # The dummy dimension allows nodes to remain unmatched
        # Final shape: (n2[b]+1, n1[b]+1, n2[b]+1, n1[b]+1)
        
        # Add dummy row (dim 0)
        K_padded = torch.cat((
            real_K, 
            torch.zeros((1, n1[b], n2[b], n1[b]), dtype=K.dtype, device=K.device)
        ), dim=0)
        
        # Add dummy column (dim 1)
        K_padded = torch.cat((
            K_padded, 
            torch.zeros((n2[b] + 1, 1, n2[b], n1[b]), dtype=K.dtype, device=K.device)
        ), dim=1)
        
        # Add dummy slice (dim 2)
        K_padded = torch.cat((
            K_padded, 
            torch.zeros((n2[b] + 1, n1[b] + 1, 1, n1[b]), dtype=K.dtype, device=K.device)
        ), dim=2)
        
        # Add dummy slice (dim 3)
        K_padded = torch.cat((
            K_padded, 
            torch.zeros((n2[b] + 1, n1[b] + 1, n2[b] + 1, 1), dtype=K.dtype, device=K.device)
        ), dim=3)
        
        # Permute to match expected format: (n1+1, n2+1, n1+1, n2+1)
        K_padded = K_padded.permute([1, 0, 3, 2])
        
        # Reshape to matrix form for A* algorithm
        padded_n1n2 = (n1[b] + 1) * (n2[b] + 1)
        K_padded = K_padded.reshape(padded_n1n2, padded_n1n2)
        
        # Run A* search for this batch
        x_pred_b = classic_astar_kernel(
            K_padded, 
            n1[b].item(), 
            n2[b].item(), 
            beam_width
        )
        
        # Remove the dummy dimension from result
        x_pred[b, :n1[b], :n2[b]] = x_pred_b[:n1[b], :n2[b]]
    
    return x_pred
