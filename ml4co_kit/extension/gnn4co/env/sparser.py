r"""
GNN4CO Sparser.
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
from typing import List, Optional
from sklearn.neighbors import KDTree
from torch_geometric.data import Data, Batch
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.utils.type_utils import to_tensor
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.task.graph.base import GraphTaskBase


class SparseData(Data):
    def __init__(
        self, 
        node_feature: Tensor, 
        edge_feature: Tensor, 
        edge_index: Tensor,
        ground_truth: Tensor
    ):
        """
        Args:
            node_feature: Node features (V, D_v)
                - The feature of each node
            edge_feature: Edge features (E, D_e)
                - The feature of each edge
            edge_index: Edge indices (2, E)
                - The indices of the start and end nodes of each edge
            ground_truth: Ground truth (V) or (E)
                - The ground truth of the graph
        """
        super(SparseData, self).__init__()
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.edge_index = edge_index
        self.ground_truth = ground_truth


class SparseDataBatch(Batch):
    """
    Custom Batch class that inherits from torch_geometric.data.Batch.
    This class explicitly declares attributes for better IDE support and type hints.
    """
    # Declare attributes for IDE autocomplete and type hints
    node_feature: Tensor     # Node features (V, D_v)
    edge_feature: Tensor     # Edge features (E, D_e)
    edge_index: Tensor       # Edge indices (2, E)
    ground_truth: Tensor     # Ground truth (V) or (E)
    batch: Tensor            # inherited from Batch, indicates which graph each node belongs to
    ptr: Optional[Tensor]    # inherited from Batch, pointer to the start of each graph

    def __init__(self, **kwargs):
        super(SparseDataBatch, self).__init__(**kwargs)

    def to_cuda(self):
        self.node_feature = self.node_feature.cuda()
        self.edge_feature = self.edge_feature.cuda()
        self.edge_index = self.edge_index.cuda()
        self.ground_truth = self.ground_truth.cuda()


class GNN4COSparser(object):
    def __init__(self, sparse_factor: int, device: str) -> None:
        self.sparse_factor = sparse_factor
        self.device = device
        self.sparse_process_func_dict = {
            TASK_TYPE.MCL: self.graph_sparse_process,
            TASK_TYPE.MCUT: self.graph_sparse_process,
            TASK_TYPE.MIS: self.graph_sparse_process,
            TASK_TYPE.MVC: self.graph_sparse_process,
            TASK_TYPE.TSP: self.tsp_sparse_process
        }
    
    def batch_data_process(
        self, batch_task_data: List[TaskBase], sampling_num: int = 1
    ) -> SparseDataBatch:
        # Preparation
        self.sparse_data_list = list()
        task_type = batch_task_data[0].task_type
        sparse_process_func = self.sparse_process_func_dict[task_type]

        # Sparse process
        for task_data in batch_task_data:
            sparse_data = sparse_process_func(task_data)
            for _ in range(sampling_num):
                self.sparse_data_list.append(sparse_data)
            
        # Merge
        batch_sparse_data = SparseDataBatch.from_data_list(self.sparse_data_list)
        if self.device == "cuda":
            batch_sparse_data.to_cuda()
        return batch_sparse_data

    def graph_sparse_process(self, task_data: GraphTaskBase) -> SparseData:
        # Nodes feature (V, 1)
        nodes_feature = to_tensor(task_data.nodes_weight).reshape(-1, 1)
        
        # Edges feature (E, 1)
        edges_feature = to_tensor(task_data.edges_weight).reshape(-1, 1) # (E, 1)
        
        # Edge index (2, E)
        edge_index = to_tensor(task_data.edge_index).float() # (2, E)
        
        # Ground Truth (V,)
        if task_data.ref_sol is not None:
            ground_truth = to_tensor(task_data.ref_sol)
        else:
            ground_truth = torch.zeros(size=(task_data.nodes_num,))
        
        # Return SparseData
        sparse_data = SparseData(
            node_feature=nodes_feature.float(),
            edge_feature=edges_feature.float(),
            edge_index=edge_index.long(),
            ground_truth=ground_truth.long()
        )
        return sparse_data

    def tsp_sparse_process(self, task_data: TSPTask) -> SparseData:
        # Nodes feature (V, 2)
        nodes_num: int = task_data.nodes_num
        points = task_data.points
        nodes_feature = to_tensor(points)

        # Edges feature (E, 1)       
        kdt = KDTree(points, leaf_size=30, metric='euclidean')
        dists_knn, idx_knn = kdt.query(points, k=self.sparse_factor, return_distance=True)
        edges_feature = to_tensor(dists_knn).reshape(-1, 1)
        
        # Edge_index (2, E)
        edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
        edge_index_0 = edge_index_0.repeat(1, self.sparse_factor).reshape(-1)
        edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
        edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        # Ground Truth (E,)
        if task_data.ref_sol is not None:
            # Reference tour
            ref_tour = task_data.ref_sol

            # Tour edges
            tour_edges = np.zeros(nodes_num, dtype=np.int64)
            tour_edges[ref_tour[:-1]] = ref_tour[1:]
            tour_edges = torch.from_numpy(tour_edges)
            tour_edges = tour_edges.reshape((-1, 1)).repeat(1, self.sparse_factor).reshape(-1)
            tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
            
            # Tour edges reverse
            tour_edges_rv = np.zeros(nodes_num, dtype=np.int64)
            tour_edges_rv[ref_tour[1:]] = ref_tour[0:-1]
            tour_edges_rv = torch.from_numpy(tour_edges_rv)
            tour_edges_rv = tour_edges_rv.reshape((-1, 1)).repeat(1, self.sparse_factor).reshape(-1)
            tour_edges_rv = torch.eq(edge_index_1, tour_edges_rv).reshape(-1, 1)
            
            # Ground truth
            ground_truth = (tour_edges + tour_edges_rv).reshape(-1).long()
        else:
            ground_truth = torch.zeros(size=(nodes_num*self.sparse_factor,))
        
        # Return SparseData
        sparse_data = SparseData(
            node_feature=nodes_feature.float(),
            edge_feature=edges_feature.float(),
            edge_index=edge_index.long(),
            ground_truth=ground_truth.long()
        )
        return sparse_data