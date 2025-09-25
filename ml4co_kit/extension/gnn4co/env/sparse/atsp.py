r"""
GNN4CO Sparse Process for ATSP.
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


def atsp_sparse_process(
    task_data: ATSPTask, sparse_factor: int
) -> Sequence[Tensor]:
    # Extract Data
    dists = task_data.dists
    ref_tour = task_data.ref_sol
    
    # nodes_num and edges_num
    nodes_num = dists.shape[0]
    edges_num = nodes_num * sparse_factor
    
    # KNN     
    idx_knn = np.argsort(dists, axis=1)[:, :sparse_factor]

    # edge_index
    edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
    edge_index_0 = edge_index_0.repeat(1, sparse_factor).reshape(-1)
    edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
    edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
    
    # edges_feature
    edges_feature_src_tgt = to_tensor(dists[edge_index_0, edge_index_1])
    edges_feature_tgt_src = to_tensor(dists[edge_index_1, edge_index_0])
    e = torch.stack([edges_feature_src_tgt, edges_feature_tgt_src], dim=1)
    
    # ground truth
    if ref_tour is not None:
        tour_edges = np.zeros(nodes_num, dtype=np.int64)
        tour_edges[ref_tour[:-1]] = ref_tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        ground_truth = torch.eq(edge_index_1, tour_edges).reshape(-1).long()
    else:
        ground_truth = None
    
    # other variables
    e = torch.zeros(size=(edges_num,))

    return (
        None, # (V,): nodes feature, ATSP does not use it
        e.float(), # (E, 2): edges feature, distance between nodes
        edge_index.long(), # (2, E): Index of edge enddists
        dists, # (V, V): Graph, full distance matrix
        ground_truth.long(), # (V,): Ground truth
        nodes_num, # Number of nodes
        edges_num # Number of edges
    )