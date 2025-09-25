r"""
Dense Process for ATSP.
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
import numpy as np
from torch import Tensor
from typing import Sequence
from ml4co_kit.task.routing.atsp import ATSPTask
from ml4co_kit.utils.type_utils import to_tensor


def atsp_dense_process(task_data: ATSPTask) -> Sequence[Tensor]:
    # Extract Data
    dists = task_data.dists
    ref_tour = task_data.ref_sol
    
    # nodes_num
    nodes_num = dists.shape[0]

    # graph
    graph = to_tensor(dists).float()
    x = torch.randn(size=(nodes_num, 2))
    
    # ground truth
    if ref_tour is not None:
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = ground_truth.long()
    else:
        ground_truth = None
    
    return (
        x, # (V, 2): nodes feature, random init
        graph.float(), # (V, V): edges feature, distance matrix
        ground_truth, # (V,): Ground truth
        nodes_num, # Number of nodes
    )