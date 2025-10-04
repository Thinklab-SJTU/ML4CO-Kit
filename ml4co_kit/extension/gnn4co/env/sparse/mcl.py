r"""
Sparse Process for MCl.
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
from torch import Tensor
from typing import Sequence
from ml4co_kit.task.graph.mcl import MClTask
from ml4co_kit.utils.type_utils import to_tensor


def mcl_sparse_process(task_data: MClTask) -> Sequence[Tensor]:
    # Ground Truth
    if task_data.ref_sol is not None:
        ground_truth = to_tensor(task_data.ref_sol)
    else:
        ground_truth = torch.zeros(size=(task_data.nodes_num,))
    
    # Return List of Tensor
    return (
        to_tensor(task_data.nodes_weight).float(),   # (V,): nodes feature
        to_tensor(task_data.edges_weight).float(),   # (E,): edges feature
        to_tensor(task_data.edge_index).long(),      # (2, E): Index of edge endpoints
        to_tensor(task_data.to_adj_matrix()).long(), # (V, V): Adjacency matrix
        ground_truth.long(),                         # (V,): Ground truth
        task_data.nodes_num,                         # Number of nodes
        task_data.edges_num                          # Number of edges
    )