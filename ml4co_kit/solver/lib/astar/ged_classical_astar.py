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
from ml4co_kit.task.graphset.ged import GEDTask
from .modules import  heuristic_prediction_hun
from ml4co_kit.solver.lib.astar.c_astar_src import c_astar 


def ged_astar(
    task_data: GEDTask,
    beam_width: int = 0
):
    """
    Pytorch implementation of ASTAR algorithm (for solving QAP)
    """
    if task_data.cost_mat is None:
        task_data.cost_mat = task_data.build_cost_mat()
     
    K = task_data.cost_mat
    K = torch.from_numpy(K)
    n1 = task_data.graphs[0].nodes_num
    if n1 is None:
        n1 = task_data.graphs[0].nodes_feature.shape[0]
    n2 = task_data.graphs[1].nodes_num
    if n2 is None:
        n2 = task_data.graphs[1].nodes_feature.shape[0]
    
    # must have n1 <= n2 for classic_astar_kernel
    if n1 > n2:
        raise ValueError('Number of nodes in graph 1 should always <= number of nodes in graph 2.')
    
    cache_dict = {}
    hun_func = functools.partial(heuristic_prediction_hun, cache_dict=cache_dict)
    x_pred, _ = c_astar(
        None,
        K, 
        n1, n2,
        None,
        hun_func,
        net_pred=False,
        beam_width=beam_width,
        trust_fact=1.,
        no_pred_size=0,
    )

    X = x_pred.detach().cpu().numpy()
    task_data.from_data(sol=X, ref=False)