r"""
AStar Solver.
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
import functools
from ml4co_kit.task.graphset.base import hungarian
from ml4co_kit.task.graphset.gm import GMTask
from .modules import  heuristic_prediction_hun
from ml4co_kit.solver.lib.astar.c_astar_src.c_astar import c_astar 
def gm_astar(
    task_data: GMTask,
    beam_width: int = 0
    ):
    """
    Pytorch implementation of ASTAR algorithm (for solving QAP)
    """
    if task_data.aff_mat is None:
        task_data.build_aff_mat()
     
    K = task_data.aff_mat
    K = torch.from_numpy(K)
    n1 = task_data.graphs[0].nodes_num
    n2 = task_data.graphs[1].nodes_num
    
    # must have n1 <= n2 for classic_astar_kernel
    if n1 > n2:
        raise ValueError('Number of nodes in graph 1 should always <= number of nodes in graph 2.')

    # output tensor
    x_pred = torch.zeros((n1, n2), device=K.device)

    # Input K is n1n2 * n1n2 but c_astar implementation requires a dummy dimension.
    # Also, n1 n2 is switched (it is column-wise vectorization in the repo, only here is row-wise vectorization)
    # The following code transforms K to fit these
    K = K.reshape(n2, n1, n2, n1)
    K_padded = torch.cat((K, torch.zeros((1, n1, n2, n1), dtype=K.dtype, device=K.device)), dim=0)
    K_padded = torch.cat((K_padded, torch.zeros((n2 + 1, 1, n2, n1), dtype=K.dtype, device=K.device)), dim=1)
    K_padded = torch.cat((K_padded, torch.zeros((n2 + 1, n1 + 1, 1, n1), dtype=K.dtype, device=K.device)), dim=2)
    K_padded = torch.cat((K_padded, torch.zeros((n2 + 1, n1 + 1, n2 + 1, 1), dtype=K.dtype, device=K.device)), dim=3)
    # K_padded shape: (n2+1) x (n1+1) x (n2+1) x (n1+1)

    K_padded = K_padded.permute([1, 0, 3, 2]) # shape: (n1+1) x (n2+1) x (n1+1) x (n2+1)
    padded_n1n2 = (n1 + 1) * (n2 + 1)
    K_padded = K_padded.reshape(padded_n1n2, padded_n1n2)
    
    cache_dict = {}
    hun_func = functools.partial(heuristic_prediction_hun, cache_dict=cache_dict)
    x_pred_b, _ = c_astar(
        None,
        -K_padded, # maximize problem -> minimize problem
        n1, n2,
        None,
        hun_func,
        net_pred=False,
        beam_width=beam_width,
        trust_fact=1.,
        no_pred_size=0,
    )
    # Remove the padded dimension
    x_pred[:, :] = x_pred_b[:n1, :n2]
    X = x_pred.detach().cpu().numpy()
    task_data.from_data(sol=X, ref=False)