r"""
Sinkhorn Algorithm.

This module implements the Sinkhorn algorithm for solving the optimal transport problem
and computing soft assignment matrices. The Sinkhorn algorithm is an iterative method
that normalizes a matrix to have specified row and column sums using matrix scaling.

The algorithm is particularly useful for graph matching problems where we need to find
a doubly stochastic matrix (soft assignment) that maximizes similarity between two graphs.
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


def pygm_sinkhorn(
    s: Tensor, 
    nrows: Tensor = None, 
    ncols: Tensor = None,
    unmatchrows: Tensor = None, 
    unmatchcols: Tensor = None,
    dummy_row: bool = False, 
    max_iter: int = 10, 
    tau: float = 1., 
    batched_operation: bool = False
) -> Tensor:
    """
    PyTorch implementation of Sinkhorn algorithm for computing soft assignment matrices.
    
    The Sinkhorn algorithm iteratively normalizes rows and columns of a matrix to make it
    doubly stochastic (rows and columns sum to 1). It operates in log-space for numerical
    stability and supports various configurations including partial matching and dummy nodes.
    
    Args:
        s: Similarity/affinity matrix of shape (batch_size, n1, n2)
           Higher values indicate better matches
        nrows: Number of valid rows for each batch, shape (batch_size,)
               If None, uses s.shape[1] for all batches
        ncols: Number of valid columns for each batch, shape (batch_size,)
               If None, uses s.shape[2] for all batches
        unmatchrows: Scores for leaving rows unmatched, shape (batch_size, n1)
                     Used for partial matching scenarios
        unmatchcols: Scores for leaving columns unmatched, shape (batch_size, n2)
                     Used for partial matching scenarios
        dummy_row: If True, adds dummy rows to make the matrix square
                   Useful when n1 < n2 and we want to match all columns
        max_iter: Maximum number of Sinkhorn iterations (default: 10)
                  More iterations lead to better convergence but higher cost
        tau: Temperature parameter for softmax (default: 1.0)
             Lower values make the assignment more discrete (harder)
             Higher values make the assignment more uniform (softer)
        batched_operation: If True, processes all batches together (faster but uses more memory)
                          If False, processes batches sequentially (slower but memory-efficient)
    
    Returns:
        Soft assignment matrix of shape (batch_size, n1, n2)
        Values are in [0, 1] and approximately doubly stochastic
    
    Note:
        - The algorithm operates in log-space for numerical stability
        - Supports rectangular matrices (n1 != n2)
        - Can handle partial matching with unmatch scores
        - Automatically transposes matrices to ensure nrows <= ncols for efficiency
    """
    batch_size = s.shape[0]
    
    # Transpose if necessary to ensure ncols >= nrows (for efficiency)
    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.transpose(1, 2)
        nrows, ncols = ncols, nrows
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True
    
    # Set default values for nrows and ncols
    if nrows is None:
        nrows = torch.tensor([s.shape[1] for _ in range(batch_size)], device=s.device)
    if ncols is None:
        ncols = torch.tensor([s.shape[2] for _ in range(batch_size)], device=s.device)
    
    # Handle batches where nrows > ncols by transposing them individually
    transposed_batch = nrows > ncols
    if torch.any(transposed_batch):
        # Transpose matrices for batches where nrows > ncols
        s_t = s.transpose(1, 2)
        s_t = torch.cat((
            s_t[:, :s.shape[1], :],
            torch.full(
                (batch_size, s.shape[1], s.shape[2] - s.shape[1]), 
                -float('inf'), 
                device=s.device
            )
        ), dim=2)
        s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, s)
        
        # Swap nrows and ncols for transposed batches
        new_nrows = torch.where(transposed_batch, ncols, nrows)
        new_ncols = torch.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols
        
        # Swap unmatch scores for transposed batches
        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = torch.cat((
                unmatchrows,
                torch.full(
                    (batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]), 
                    -float('inf'), 
                    device=s.device
                )
            ), dim=1)
            new_unmatchrows = torch.where(
                transposed_batch.view(batch_size, 1), 
                unmatchcols, 
                unmatchrows_pad
            )[:, :unmatchrows.shape[1]]
            new_unmatchcols = torch.where(
                transposed_batch.view(batch_size, 1), 
                unmatchrows_pad, 
                unmatchcols
            )
            unmatchrows = new_unmatchrows
            unmatchcols = new_unmatchcols
    
    # Convert to log-space and apply temperature scaling
    log_s = s / tau
    if unmatchrows is not None and unmatchcols is not None:
        unmatchrows = unmatchrows / tau
        unmatchcols = unmatchcols / tau
    
    # Add dummy rows if requested (to make matrix square)
    if dummy_row:
        if not log_s.shape[2] >= log_s.shape[1]:
            raise RuntimeError('Error in Sinkhorn with dummy row: ncols must be >= nrows')
        
        dummy_shape = list(log_s.shape)
        dummy_shape[1] = log_s.shape[2] - log_s.shape[1]
        ori_nrows = nrows
        nrows = ncols.clone()
        
        # Concatenate dummy rows filled with -inf
        log_s = torch.cat((
            log_s, 
            torch.full(dummy_shape, -float('inf'), device=log_s.device, dtype=log_s.dtype)
        ), dim=1)
        
        if unmatchrows is not None:
            unmatchrows = torch.cat((
                unmatchrows,
                torch.full(
                    (dummy_shape[0], dummy_shape[1]), 
                    -float('inf'), 
                    device=log_s.device,
                    dtype=log_s.dtype
                )
            ), dim=1)
        
        # Set dummy row values to a large negative number
        for b in range(batch_size):
            log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
    
    # Augment matrix for partial matching (add unmatch row/column)
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = torch.full(
            (log_s.shape[0], log_s.shape[1] + 1, log_s.shape[2] + 1), 
            -float('inf'),
            device=log_s.device, 
            dtype=log_s.dtype
        )
        new_log_s[:, :-1, :-1] = log_s
        log_s = new_log_s
        
        # Fill unmatch row and column with unmatch scores
        for b in range(batch_size):
            log_s[b, :nrows[b], ncols[b]] = unmatchrows[b, :nrows[b]]
            log_s[b, nrows[b], :ncols[b]] = unmatchcols[b, :ncols[b]]
    
    # Create masks for valid rows and columns
    row_mask = torch.zeros(batch_size, log_s.shape[1], 1, dtype=torch.bool, device=log_s.device)
    col_mask = torch.zeros(batch_size, 1, log_s.shape[2], dtype=torch.bool, device=log_s.device)
    for b in range(batch_size):
        row_mask[b, :nrows[b], 0] = 1
        col_mask[b, 0, :ncols[b]] = 1
    
    if unmatchrows is not None and unmatchcols is not None:
        ncols += 1
        nrows += 1
    
    # Perform Sinkhorn iterations
    if batched_operation:
        # Batched processing: all batches together
        # Mask out invalid entries
        for b in range(batch_size):
            log_s[b, nrows[b]:, :] = -float('inf')
            log_s[b, :, ncols[b]:] = -float('inf')
        
        # Alternating row and column normalization
        for i in range(max_iter):
            if i % 2 == 0:
                # Normalize rows: make each row sum to 1 in probability space
                log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                log_s = log_s - torch.where(row_mask, log_sum, torch.zeros_like(log_sum))
                if torch.any(torch.isnan(log_s)):
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')
            else:
                # Normalize columns: make each column sum to 1 in probability space
                log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                log_s = log_s - torch.where(col_mask, log_sum, torch.zeros_like(log_sum))
                if torch.any(torch.isnan(log_s)):
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')
        
        ret_log_s = log_s
    else:
        # Sequential processing: process each batch separately (more memory-efficient)
        ret_log_s = torch.full(
            (batch_size, log_s.shape[1], log_s.shape[2]), 
            -float('inf'), 
            device=log_s.device,
            dtype=log_s.dtype
        )
        
        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]
            
            # Alternating row and column normalization for this batch
            for i in range(max_iter):
                if i % 2 == 0:
                    # Normalize rows
                    log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                    log_s_b = log_s_b - torch.where(row_mask_b, log_sum, torch.zeros_like(log_sum))
                else:
                    # Normalize columns
                    log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                    log_s_b = log_s_b - torch.where(col_mask_b, log_sum, torch.zeros_like(log_sum))
            
            ret_log_s[b, row_slice, col_slice] = log_s_b
    
    # Remove unmatch row and column if they were added
    if unmatchrows is not None and unmatchcols is not None:
        ncols -= 1
        nrows -= 1
        for b in range(batch_size):
            ret_log_s[b, :nrows[b] + 1, ncols[b]] = -float('inf')
            ret_log_s[b, nrows[b], :ncols[b]] = -float('inf')
        ret_log_s = ret_log_s[:, :-1, :-1]
    
    # Remove dummy rows if they were added
    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        for b in range(batch_size):
            ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')
    
    # Transpose back batches that were transposed
    if torch.any(transposed_batch):
        s_t = ret_log_s.transpose(1, 2)
        s_t = torch.cat((
            s_t[:, :ret_log_s.shape[1], :],
            torch.full(
                (batch_size, ret_log_s.shape[1], ret_log_s.shape[2] - ret_log_s.shape[1]), 
                -float('inf'),
                device=log_s.device
            )
        ), dim=2)
        ret_log_s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, ret_log_s)
    
    # Transpose back if the input was transposed
    if transposed:
        ret_log_s = ret_log_s.transpose(1, 2)
    
    # Convert from log-space back to probability space
    return torch.exp(ret_log_s)
