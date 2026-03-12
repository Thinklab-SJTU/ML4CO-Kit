r"""
Spectral Matching (SM) Algorithm.

This module implements the Spectral Matching algorithm for graph matching.
SM is based on the power iteration method to find the principal eigenvector
of the affinity matrix, which corresponds to the optimal matching.

The algorithm iteratively multiplies the affinity matrix with the current
solution and normalizes, converging to the dominant eigenvector.

Reference:
    Leordeanu, M., & Hebert, M. (2005). A spectral technique for correspondence
    problems using pairwise constraints. In ICCV 2005.
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
from .utils import _check_and_init_gm


def pygm_sm(
    K: Tensor, 
    n1: Tensor, 
    n2: Tensor, 
    n1max: int, 
    n2max: int, 
    x0: Tensor,
    max_iter: int
) -> Tensor:
    """
    PyTorch implementation of Spectral Matching (SM) algorithm.
    
    SM solves the graph matching problem by finding the principal eigenvector
    of the affinity matrix using power iteration. The algorithm is based on
    the observation that the optimal matching corresponds to the eigenvector
    associated with the largest eigenvalue of the affinity matrix.
    
    The power iteration method:
    1. Multiply: v_new = K * v_old
    2. Normalize: v_new = v_new / ||v_new||_2
    3. Repeat until convergence
    
    Args:
        K: Affinity matrix of shape (batch_size, n2max*n1max, n2max*n1max)
           K[b, j*n1max+i, l*n1max+k] represents the affinity between 
           matching (i,j) and matching (k,l) in batch b
        n1: Number of nodes in graph 1 for each batch, shape (batch_size,)
            If None, uses n1max for all batches
        n2: Number of nodes in graph 2 for each batch, shape (batch_size,)
            If None, uses n2max for all batches
        n1max: Maximum number of nodes in graph 1 across all batches
        n2max: Maximum number of nodes in graph 2 across all batches
        x0: Initial solution of shape (batch_size, n1max, n2max)
            If None, initializes with uniform distribution
        max_iter: Maximum number of power iterations
    
    Returns:
        Soft assignment matrix of shape (batch_size, n1max, n2max)
        Values represent the principal eigenvector reshaped as a matrix

    Note:
        - This is a continuous relaxation method (returns soft assignments)
        - Use Hungarian algorithm for post-processing if discrete solution is needed
        - Convergence is detected when L2 norm change is less than 1e-5
        - The method is fast and simple but may get stuck in local optima
    """
    # Initialize parameters and variables
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    
    v = vlast = v0
    
    # Power iteration loop
    for i in range(max_iter):
        # Step 1: Multiply with affinity matrix
        # This is equivalent to: v_new = K * v_old
        v = torch.bmm(K, v)
        
        # Step 2: Normalize using L2 norm
        # This ensures the vector stays on the unit sphere
        n = torch.norm(v, p=2, dim=1)
        v = torch.matmul(v, (1 / n).view(batch_num, 1, 1))
        
        # Check convergence: stop if change is very small
        if torch.norm(v - vlast) < 1e-5:
            break
        
        vlast = v
    
    # Reshape from vector to assignment matrix format
    x = v.view(batch_num, n2max, n1max).transpose(1, 2)
    
    return x
