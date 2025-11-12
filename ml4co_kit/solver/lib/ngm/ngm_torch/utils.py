r"""
Utils for NGM
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import scipy
import numpy as np
from torch import Tensor
from multiprocessing import Pool



def _check_and_init_gm(K: Tensor, n1: Tensor = None, n2: Tensor = None, n1max: Tensor = None, n2max: Tensor = None, x0: Tensor = None):
    # get batch number
    batch_num = K.shape[0]
    n1n2 = K.shape[1]

    # get values of n1, n2, n1max, n2max and check
    if n1 is None:
        n1 = torch.full((batch_num,), n1max, dtype=torch.int, device=K.device)
    elif type(n1) is Tensor and len(n1.shape) == 0:
        n1 = n1.unsqueeze(0)
    if n2 is None:
        n2 = torch.full((batch_num,), n2max, dtype=torch.int, device=K.device)
    elif type(n2) is Tensor and len(n2.shape) == 0:
        n2 = n2.unsqueeze(0)
    if n1max is None:
        n1max = torch.max(n1)
    if n2max is None:
        n2max = torch.max(n2)

    if not n1max * n2max == n1n2:
        raise ValueError('the input size of K does not match with n1max * n2max!')

    # initialize x0 (also v0)
    if x0 is None:
        x0 = torch.zeros(batch_num, n1max, n2max, dtype=K.dtype, device=K.device)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = torch.tensor(1.) / (n1[b] * n2[b])
    v0 = x0.transpose(1, 2).reshape(batch_num, n1n2, 1)

    return batch_num, n1, n2, n1max, n2max, n1n2, v0

def sinkhorn(s: Tensor, nrows: Tensor = None, ncols: Tensor = None,
             unmatchrows: Tensor = None, unmatchcols: Tensor = None,
             dummy_row: bool = False, max_iter: int = 10, tau: float = 1., batched_operation: bool = False) -> Tensor:
    """
    Pytorch implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.transpose(1, 2)
        nrows, ncols = ncols, nrows
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True

    if nrows is None:
        nrows = torch.tensor([s.shape[1] for _ in range(batch_size)], device=s.device)
    if ncols is None:
        ncols = torch.tensor([s.shape[2] for _ in range(batch_size)], device=s.device)

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if torch.any(transposed_batch):
        s_t = s.transpose(1, 2)
        s_t = torch.cat((
            s_t[:, :s.shape[1], :],
            torch.full((batch_size, s.shape[1], s.shape[2] - s.shape[1]), -float('inf'), device=s.device)), dim=2)
        s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, s)

        new_nrows = torch.where(transposed_batch, ncols, nrows)
        new_ncols = torch.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = torch.cat((
                unmatchrows,
                torch.full((batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]), -float('inf'), device=s.device)),
                dim=1)
            new_unmatchrows = torch.where(transposed_batch.view(batch_size, 1), unmatchcols, unmatchrows_pad)[:,
                              :unmatchrows.shape[1]]
            new_unmatchcols = torch.where(transposed_batch.view(batch_size, 1), unmatchrows_pad, unmatchcols)
            unmatchrows = new_unmatchrows
            unmatchcols = new_unmatchcols

    # operations are performed on log_s
    log_s = s / tau
    if unmatchrows is not None and unmatchcols is not None:
        unmatchrows = unmatchrows / tau
        unmatchcols = unmatchcols / tau

    if dummy_row:
        if not log_s.shape[2] >= log_s.shape[1]:
            raise RuntimeError('Error in Sinkhorn with dummy row')
        dummy_shape = list(log_s.shape)
        dummy_shape[1] = log_s.shape[2] - log_s.shape[1]
        ori_nrows = nrows
        nrows = ncols.clone()
        log_s = torch.cat((log_s, torch.full(dummy_shape, -float('inf'), device=log_s.device, dtype=log_s.dtype)),
                          dim=1)
        if unmatchrows is not None:
            unmatchrows = torch.cat((unmatchrows,
                                     torch.full((dummy_shape[0], dummy_shape[1]), -float('inf'), device=log_s.device,
                                                dtype=log_s.dtype)), dim=1)
        for b in range(batch_size):
            log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100

    # assign the unmatch weights
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = torch.full((log_s.shape[0], log_s.shape[1] + 1, log_s.shape[2] + 1), -float('inf'),
                               device=log_s.device, dtype=log_s.dtype)
        new_log_s[:, :-1, :-1] = log_s
        log_s = new_log_s
        for b in range(batch_size):
            log_s[b, :nrows[b], ncols[b]] = unmatchrows[b, :nrows[b]]
            log_s[b, nrows[b], :ncols[b]] = unmatchcols[b, :ncols[b]]
    row_mask = torch.zeros(batch_size, log_s.shape[1], 1, dtype=torch.bool, device=log_s.device)
    col_mask = torch.zeros(batch_size, 1, log_s.shape[2], dtype=torch.bool, device=log_s.device)
    for b in range(batch_size):
        row_mask[b, :nrows[b], 0] = 1
        col_mask[b, 0, :ncols[b]] = 1
    if unmatchrows is not None and unmatchcols is not None:
        ncols += 1
        nrows += 1

    if batched_operation:
        for b in range(batch_size):
            log_s[b, nrows[b]:, :] = -float('inf')
            log_s[b, :, ncols[b]:] = -float('inf')

        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                log_s = log_s - torch.where(row_mask, log_sum, torch.zeros_like(log_sum))
                if torch.any(torch.isnan(log_s)):
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')
            else:
                log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                log_s = log_s - torch.where(col_mask, log_sum, torch.zeros_like(log_sum))
                if torch.any(torch.isnan(log_s)):
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')

        ret_log_s = log_s
    else:
        ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=log_s.device,
                               dtype=log_s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                    log_s_b = log_s_b - torch.where(row_mask_b, log_sum, torch.zeros_like(log_sum))
                else:
                    log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                    log_s_b = log_s_b - torch.where(col_mask_b, log_sum, torch.zeros_like(log_sum))

            ret_log_s[b, row_slice, col_slice] = log_s_b

    if unmatchrows is not None and unmatchcols is not None:
        ncols -= 1
        nrows -= 1
        for b in range(batch_size):
            ret_log_s[b, :nrows[b] + 1, ncols[b]] = -float('inf')
            ret_log_s[b, nrows[b], :ncols[b]] = -float('inf')
        ret_log_s = ret_log_s[:, :-1, :-1]

    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        for b in range(batch_size):
            ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

    if torch.any(transposed_batch):
        s_t = ret_log_s.transpose(1, 2)
        s_t = torch.cat((
            s_t[:, :ret_log_s.shape[1], :],
            torch.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2] - ret_log_s.shape[1]), -float('inf'),
                       device=log_s.device)), dim=2)
        ret_log_s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, ret_log_s)

    if transposed:
        ret_log_s = ret_log_s.transpose(1, 2)

    return torch.exp(ret_log_s)

def _load_model(model, path, device, strict=True):
    """
    Load PyTorch model from a given path. strict=True means all keys must be matched
    """
    if isinstance(model, torch.nn.DataParallel):
        module = model.module
    else:
        module = model
    missing_keys, unexpected_keys = module.load_state_dict(torch.load(path, map_location=device), strict=strict)
    if len(unexpected_keys) > 0:
        print('Warning: Unexpected key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        print('Warning: Missing key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in missing_keys)))
