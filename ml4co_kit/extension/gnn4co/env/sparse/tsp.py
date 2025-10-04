r"""
GNN4CO Sparse Process for TSP.
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
from sklearn.neighbors import KDTree
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.utils.type_utils import to_tensor


def tsp_sparse_process(task_data: TSPTask, sparse_factor: int) -> Sequence[Tensor]:
    # Extract Data
    points = task_data.points
    ref_tour = task_data.ref_sol
    
    # nodes_num and edges_num
    nodes_num = points.shape[0]
    edges_num = nodes_num * sparse_factor
    
    # KDTree        
    kdt = KDTree(points, leaf_size=30, metric='euclidean')
    dists_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)
    e = to_tensor(dists_knn.reshape(-1))
    
    # edge_index
    edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
    edge_index_0 = edge_index_0.repeat(1, sparse_factor).reshape(-1)
    edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
    edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)
    
    # ground truth
    if ref_tour is not None:
        tour_edges = np.zeros(nodes_num, dtype=np.int64)
        tour_edges[ref_tour[:-1]] = ref_tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
        
        tour_edges_rv = np.zeros(nodes_num, dtype=np.int64)
        tour_edges_rv[ref_tour[1:]] = ref_tour[0:-1]
        tour_edges_rv = torch.from_numpy(tour_edges_rv)
        tour_edges_rv = tour_edges_rv.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        tour_edges_rv = torch.eq(edge_index_1, tour_edges_rv).reshape(-1, 1)
        ground_truth = (tour_edges + tour_edges_rv).reshape(-1).long()
    else:
        ground_truth = None
    
    # nodes feature
    x = to_tensor(points)
    
    return (
        x.float(), # (V, 2): nodes feature, Euler coordinates of nodes
        e.float(), # (E,): edges feature, distance between nodes
        edge_index.long(), # (2, E): Index of edge endpoints
        None, # (V, V): Graph, but no need for TSP when sparse
        ground_truth, # (V,): Ground truth
        nodes_num, # Number of nodes
        edges_num # Number of edges
    )