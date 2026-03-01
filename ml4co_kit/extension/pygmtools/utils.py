r"""
Utility Functions for Graph Matching Algorithms.

This module provides helper functions used by various graph matching algorithms,
including parameter validation and initialization routines.
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
from typing import Tuple, Optional


def _check_and_init_gm(
    K: Tensor,
    n1: Optional[Tensor],
    n2: Optional[Tensor],
    n1max: Optional[int],
    n2max: Optional[int],
    x0: Optional[Tensor]
) -> Tuple[int, Tensor, Tensor, int, int, int, Tensor]:
    """
    Check parameters and initialize variables for graph matching algorithms.
    
    This function performs the following tasks:
    1. Validates input dimensions and consistency
    2. Sets default values for optional parameters
    3. Initializes the starting solution if not provided
    
    Args:
        K: Affinity matrix of shape (batch_size, n1max*n2max, n1max*n2max)
           Encodes pairwise affinities between potential matches
        n1: Number of nodes in graph 1 for each batch, shape (batch_size,)
            Can be None (uses n1max) or scalar (broadcasts to all batches)
        n2: Number of nodes in graph 2 for each batch, shape (batch_size,)
            Can be None (uses n2max) or scalar (broadcasts to all batches)
        n1max: Maximum number of nodes in graph 1 across all batches
               If None, computed as max(n1)
        n2max: Maximum number of nodes in graph 2 across all batches
               If None, computed as max(n2)
        x0: Initial solution of shape (batch_size, n1max, n2max)
            If None, initializes with uniform distribution over valid entries
    
    Returns:
        Tuple containing:
        - batch_num: Number of batches (int)
        - n1: Number of nodes in graph 1, shape (batch_size,)
        - n2: Number of nodes in graph 2, shape (batch_size,)
        - n1max: Maximum nodes in graph 1 (int)
        - n2max: Maximum nodes in graph 2 (int)
        - n1n2: Product n1max * n2max (int)
        - v0: Initial solution in vector form, shape (batch_size, n1max*n2max, 1)
    
    Raises:
        ValueError: If K.shape[1] != n1max * n2max (dimension mismatch)
    """
    # Get batch number from affinity matrix
    batch_num = K.shape[0]
    n1n2 = K.shape[1]
    
    # Process n1: number of nodes in graph 1
    if n1 is None:
        # If not provided, assume all batches have n1max nodes
        n1 = torch.full((batch_num,), n1max, dtype=torch.int32, device=K.device)
    elif type(n1) is Tensor and len(n1.shape) == 0:
        # If scalar tensor, unsqueeze to make it 1D
        n1 = n1.unsqueeze(0)
    
    # Process n2: number of nodes in graph 2
    if n2 is None:
        # If not provided, assume all batches have n2max nodes
        n2 = torch.full((batch_num,), n2max, dtype=torch.int32, device=K.device)
    elif type(n2) is Tensor and len(n2.shape) == 0:
        # If scalar tensor, unsqueeze to make it 1D
        n2 = n2.unsqueeze(0)
    
    # Compute n1max and n2max if not provided
    if n1max is None:
        n1max = torch.max(n1)
    if n2max is None:
        n2max = torch.max(n2)
    
    # Validate dimensions
    if not n1max * n2max == n1n2:
        raise ValueError(
            f'The input size of K ({n1n2}) does not match with n1max * n2max ({n1max * n2max})!'
        )
    
    # Initialize starting solution x0 (and its vectorized form v0)
    if x0 is None:
        # Create zero matrix
        x0 = torch.zeros(batch_num, n1max, n2max, dtype=K.dtype, device=K.device)
        
        # Initialize valid entries with uniform distribution
        # For each batch, set x0[b, 0:n1[b], 0:n2[b]] to 1/(n1[b]*n2[b])
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = torch.tensor(1.) / (n1[b] * n2[b])
    
    # Convert assignment matrix to vector form
    # Transpose and reshape: (batch, n1max, n2max) -> (batch, n2max, n1max) -> (batch, n1n2, 1)
    v0 = x0.transpose(1, 2).reshape(batch_num, n1n2, 1)
    
    return batch_num, n1, n2, n1max, n2max, n1n2, v0
