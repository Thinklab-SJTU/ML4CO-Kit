r"""
Hungarian Algorithm.

This module implements the Hungarian algorithm (also known as the Kuhn-Munkres algorithm)
for solving the linear assignment problem. It finds the optimal one-to-one matching between
two sets that minimizes the total cost.
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
import numpy as np
import scipy.optimize
from torch import Tensor
from multiprocessing import Pool


def _hung_kernel(
    s: np.ndarray, 
    n1: int = None, 
    n2: int = None, 
    unmatch1: np.ndarray = None, 
    unmatch2: np.ndarray = None
) -> np.ndarray:
    """
    Hungarian kernel function by calling the linear sum assignment solver from Scipy.
    
    This function solves the linear assignment problem using the Hungarian algorithm.
    It supports partial matching by allowing unmatched nodes with specified costs.
    
    Args:
        s: Cost matrix of shape (n1, n2), where s[i,j] is the cost of matching node i to node j
        n1: Number of nodes in the first set (default: s.shape[0])
        n2: Number of nodes in the second set (default: s.shape[1])
        unmatch1: Cost of leaving nodes in set 1 unmatched, shape (n1,)
        unmatch2: Cost of leaving nodes in set 2 unmatched, shape (n2,)
    
    Returns:
        perm_mat: Permutation matrix of shape (n1, n2), where perm_mat[i,j]=1 means node i matches node j
    """
    # Set default values for n1 and n2
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    
    # Handle partial matching case
    if unmatch1 is not None and unmatch2 is not None:
        # Construct augmented cost matrix to allow unmatched nodes
        # Structure:
        #   [upper_left  | upper_right]
        #   [lower_left  | lower_right]
        # where upper_left is the original cost matrix,
        # and diagonal elements in upper_right/lower_left represent unmatch costs
        
        upper_left = s[:n1, :n2]
        
        # Upper right: cost for leaving nodes in set 1 unmatched
        upper_right = np.full((n1, n1), float('inf'))
        np.fill_diagonal(upper_right, unmatch1[:n1])
        
        # Lower left: cost for leaving nodes in set 2 unmatched
        lower_left = np.full((n2, n2), float('inf'))
        np.fill_diagonal(lower_left, unmatch2[:n2])
        
        # Lower right: dummy nodes (zero cost)
        lower_right = np.zeros((n2, n1))
        
        # Concatenate to form the augmented cost matrix
        large_cost_mat = np.concatenate(
            (
                np.concatenate((upper_left, upper_right), axis=1),
                np.concatenate((lower_left, lower_right), axis=1)
            ), 
            axis=0
        )
        
        # Solve the augmented assignment problem
        row, col = scipy.optimize.linear_sum_assignment(large_cost_mat)
        
        # Filter out dummy matches (only keep valid matches)
        valid_idx = np.logical_and(row < n1, col < n2)
        row = row[valid_idx]
        col = col[valid_idx]
    else:
        # Standard assignment problem without partial matching
        row, col = scipy.optimize.linear_sum_assignment(s[:n1, :n2])
    
    # Construct permutation matrix from row and column indices
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    
    return perm_mat


def pygm_hungarian(
    s: Tensor, 
    n1: Tensor = None, 
    n2: Tensor = None,
    unmatch1: Tensor = None, 
    unmatch2: Tensor = None,
    nproc: int = 1
) -> Tensor:
    """
    PyTorch implementation of Hungarian algorithm with batch processing support.
    
    This function solves the linear assignment problem for batched cost matrices.
    It can utilize multiple processes for parallel computation.
    
    Args:
        s: Batched cost matrix of shape (batch_size, n1, n2)
           Note: This is a similarity matrix, will be negated to cost matrix internally
        n1: Number of nodes in the first set for each batch, shape (batch_size,)
            If None, uses s.shape[1] for all batches
        n2: Number of nodes in the second set for each batch, shape (batch_size,)
            If None, uses s.shape[2] for all batches
        unmatch1: Cost of leaving nodes in set 1 unmatched, shape (batch_size, n1)
                  Note: This is a similarity value, will be negated internally
        unmatch2: Cost of leaving nodes in set 2 unmatched, shape (batch_size, n2)
                  Note: This is a similarity value, will be negated internally
        nproc: Number of parallel processes to use (default: 1)
    
    Returns:
        perm_mat: Batched permutation matrices of shape (batch_size, n1, n2)
    """
    device = s.device
    batch_num = s.shape[0]
    
    # Convert similarity matrix to cost matrix (negate values)
    perm_mat = s.cpu().detach().numpy() * -1
    
    # Process n1: number of nodes in first set
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    
    # Process n2: number of nodes in second set
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num
    
    # Process unmatch costs (negate similarity to cost)
    if unmatch1 is not None:
        unmatch1 = -unmatch1.cpu().numpy()
    else:
        unmatch1 = [None] * batch_num
    
    if unmatch2 is not None:
        unmatch2 = -unmatch2.cpu().numpy()
    else:
        unmatch2 = [None] * batch_num
    
    # Solve assignment problem for each batch
    if nproc > 1:
        # Parallel processing with multiple processes
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(
                _hung_kernel, 
                zip(perm_mat, n1, n2, unmatch1, unmatch2)
            )
            perm_mat = np.stack(mapresult.get())
    else:
        # Sequential processing
        perm_mat = np.stack([
            _hung_kernel(perm_mat[b], n1[b], n2[b], unmatch1[b], unmatch2[b]) 
            for b in range(batch_num)
        ])
    
    # Convert back to PyTorch tensor and move to original device
    perm_mat = torch.from_numpy(perm_mat).to(device)
    
    return perm_mat