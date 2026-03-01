r"""
Integer Projected Fixed Point (IPFP) Algorithm.

This module implements the IPFP algorithm for graph matching. IPFP is an iterative
algorithm that alternates between:
1. Computing a continuous relaxation by moving along the gradient direction
2. Projecting to the nearest integer solution using the Hungarian algorithm

The algorithm is particularly effective for quadratic assignment problems (QAP)
and graph matching tasks.

Reference:
    Leordeanu, M., & Hebert, M. (2006). Efficient MAP approximation for dense energy functions.
    In ICML 2006.
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
from torch import Tensor
from .hungarian import pygm_hungarian
from .utils import _check_and_init_gm


def pygm_ipfp(
    K: Tensor, 
    n1: Tensor, 
    n2: Tensor, 
    n1max: int, 
    n2max: int, 
    x0: Tensor,
    max_iter: int
) -> Tensor:
    """
    PyTorch implementation of Integer Projected Fixed Point (IPFP) algorithm.
    
    IPFP solves the quadratic assignment problem by iteratively:
    1. Computing the gradient direction in the continuous space
    2. Finding the optimal step size along this direction
    3. Projecting to the nearest discrete solution using Hungarian algorithm
    
    The algorithm maintains the best solution found so far and terminates when
    convergence is detected or maximum iterations are reached.
    
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
        x0: Initial solution of shape (batch_size, n1max, n2max)
            If None, initializes with uniform distribution
        max_iter: Maximum number of iterations
    
    Returns:
        Predicted assignment matrix of shape (batch_size, n1max, n2max)
        Values are binary (0 or 1) representing the matching

    Note:
        - The algorithm finds a local optimum, not necessarily global
        - Convergence is detected when objective change is less than 0.1%
        - The best solution across all iterations is returned
    """
    # Initialize parameters and variables
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v
    best_v = v
    best_obj = -1
    
    def comp_obj_score(v1: Tensor, K: Tensor, v2: Tensor) -> Tensor:
        """
        Compute objective score: v1^T * K * v2
        
        Args:
            v1: First vector of shape (batch_size, n1n2, 1)
            K: Affinity matrix of shape (batch_size, n1n2, n1n2)
            v2: Second vector of shape (batch_size, n1n2, 1)
        
        Returns:
            Objective score of shape (batch_size, 1, 1)
        """
        return torch.bmm(torch.bmm(v1.view(batch_num, 1, -1), K), v2)
    
    # Main IPFP iteration loop
    for i in range(max_iter):
        # Step 1: Compute gradient direction
        # cost = K * v represents the gradient of the objective
        cost = torch.bmm(K, v).reshape(batch_num, n2max, n1max).transpose(1, 2)
        
        # Step 2: Project to discrete solution using Hungarian algorithm
        binary_sol = pygm_hungarian(cost, n1, n2)
        binary_v = binary_sol.transpose(1, 2).view(batch_num, -1, 1)
        
        # Step 3: Line search to find optimal step size
        # We want to maximize: (v + t*(binary_v - v))^T * K * (v + t*(binary_v - v))
        # Taking derivative w.r.t. t and setting to 0:
        # t* = -alpha / beta where:
        alpha = comp_obj_score(v, K, binary_v - v)
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = -alpha / beta
        
        # Update v with optimal step size (clipped to [0, 1])
        v = torch.where(
            torch.logical_or(beta >= 0, t0 >= 1), 
            binary_v,  # If beta >= 0 or t0 >= 1, jump directly to binary solution
            v + t0 * (binary_v - v)  # Otherwise, take optimal step
        )
        
        # Compute objective scores
        last_v_obj = comp_obj_score(last_v, K, last_v)
        current_obj = comp_obj_score(binary_v, K, binary_v)
        
        # Update best solution
        best_v = torch.where(current_obj > best_obj, binary_v, best_v)
        best_obj = torch.where(current_obj > best_obj, current_obj, best_obj)
        
        # Check convergence: stop if objective change is less than 0.1%
        if torch.max(torch.abs(last_v_obj - current_obj) / last_v_obj) < 1e-3:
            break
        
        last_v = v
    
    # Reshape best solution to assignment matrix format
    pred_x = best_v.reshape(batch_num, n2max, n1max).transpose(1, 2)
    
    return pred_x
