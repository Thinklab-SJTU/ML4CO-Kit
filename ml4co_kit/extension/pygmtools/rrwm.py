r"""
Reweighted Random Walk Matching (RRWM) Algorithm.

This module implements the RRWM algorithm for graph matching. RRWM combines
random walk and reweighting strategies to find correspondences between graphs.

The algorithm alternates between:
1. Random walk: propagating matching scores through the affinity graph
2. Reweighted jump: normalizing and reweighting the solution using Sinkhorn

This combination helps escape local optima and find better matchings.

Reference:
    Cho, M., Lee, J., & Lee, K. M. (2010). Reweighted random walks for graph matching.
    In ECCV 2010.
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
from .sinkhorn import pygm_sinkhorn
from .utils import _check_and_init_gm


def pygm_rrwm(
    K: Tensor, 
    n1: Tensor, 
    n2: Tensor, 
    n1max: int, 
    n2max: int, 
    x0: Tensor,
    max_iter: int, 
    sk_iter: int, 
    alpha: float, 
    beta: float
) -> Tensor:
    """
    PyTorch implementation of Reweighted Random Walk Matching (RRWM) algorithm.
    
    RRWM performs graph matching by iteratively:
    1. Random walk: multiply the current solution with the affinity matrix
    2. Normalization: normalize to maintain probability distribution
    3. Reweighted jump: apply Sinkhorn normalization with reweighting
    
    The algorithm combines the exploration capability of random walks with
    the optimization power of reweighting to find high-quality matchings.
    
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
        max_iter: Maximum number of outer iterations (random walk + reweighting)
        sk_iter: Number of Sinkhorn iterations in the reweighting step
        alpha: Weight for the reweighted solution (typically in [0, 1])
               alpha=1 means only use reweighted solution
               alpha=0 means only use random walk solution
        beta: Scaling factor for reweighting (typically > 0)
              Higher values make the reweighting more aggressive
    
    Returns:
        Soft assignment matrix of shape (batch_size, n1max, n2max)
        Values are in [0, 1] representing matching probabilities

    Note:
        - Convergence is detected when L1 norm change is less than 1e-5
        - The algorithm returns a soft assignment (not binary)
        - Use Hungarian algorithm for post-processing if discrete solution is needed
    """
    # Initialize parameters and variables
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    
    # Rescale the affinity matrix for numerical stability
    # Normalize by the maximum row sum to prevent overflow
    d = K.sum(dim=2, keepdim=True)
    dmax = d.max(dim=1, keepdim=True).values
    K = K / (dmax + d.min() * 1e-5)
    v = v0
    
    # Main RRWM iteration loop
    for i in range(max_iter):
        # Step 1: Random walk
        # Propagate matching scores through the affinity graph
        v = torch.bmm(K, v)
        last_v = v
        
        # Normalize to maintain probability distribution (L1 norm = 1)
        n = torch.norm(v, p=1, dim=1, keepdim=True)
        v = v / n
        
        # Step 2: Reweighted jump
        # Reshape to assignment matrix format
        s = v.view(batch_num, n2max, n1max).transpose(1, 2)
        
        # Scale by beta and normalize by maximum value
        s = beta * s / s.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
        
        # Apply Sinkhorn normalization to get doubly stochastic matrix
        s_normalized = pygm_sinkhorn(s, n1, n2, max_iter=sk_iter, batched_operation=True)
        
        # Combine random walk result with reweighted result
        # alpha controls the balance between exploration (random walk) and exploitation (reweighting)
        v = alpha * s_normalized.transpose(1, 2).reshape(batch_num, n1n2, 1) + \
            (1 - alpha) * v
        
        # Normalize again to maintain probability distribution
        n = torch.norm(v, p=1, dim=1, keepdim=True)
        v = torch.matmul(v, 1 / n)
        
        # Check convergence: stop if change is very small
        if torch.norm(v - last_v) < 1e-5:
            break
    
    # Reshape to assignment matrix format and return
    return v.view(batch_num, n2max, n1max).transpose(1, 2)
