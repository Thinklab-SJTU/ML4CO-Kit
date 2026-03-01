r""""
Graph Edit Distance

Graph Edit Distance aim to find both the minimum total cost and the corresponding edit path 
required to transform graph G₁ into graph G₂ through a sequence of basic edit operations,
including node/edge insertion, node/edge deletion, and node/edge substitution.
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


import numpy as np
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.base import QAPTaskBase, QAPGraphBase


class GEDTask(QAPTaskBase):
    def __init__(
        self,
        node_sub_cost: float = 1.0,
        node_ins_cost: float = 1.0,
        node_del_cost: float = 1.0,
        edge_sub_cost: float = 1.0,
        edge_ins_cost: float = 1.0,
        edge_del_cost: float = 1.0,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(GEDTask, self).__init__(
            task_type=TASK_TYPE.GED,
            minimize=False,
            precision=precision
        )
        
        # Initialize Attributes (Cost)
        self.node_sub_cost = node_sub_cost
        self.node_ins_cost = node_ins_cost
        self.node_del_cost = node_del_cost
        self.edge_sub_cost = edge_sub_cost
        self.edge_ins_cost = edge_ins_cost
        self.edge_del_cost = edge_del_cost

        # Initialize Attributes (Graph)
        self.g1: QAPGraphBase = None
        self.g2: QAPGraphBase = None

        # Initialize Attributes (Threshold)
        self.threshold = threshold
  
    def build_cost_mat(self) -> np.ndarray:
        """Build cost matrix from node and edge features."""
        # Get adjacency matrices
        n1 = self.g1.nodes_num
        n2 = self.g2.nodes_num
        n1_plus_1 = n1 + 1
        n2_plus_1 = n2 + 1
        adj_1 = self.g1.to_adj_matrix()
        adj_2 = self.g2.to_adj_matrix()

        # Get dummy adjacency matrices
        dummy_adj_1 = np.zeros((n1_plus_1, n1_plus_1), dtype=np.int32)
        dummy_adj_2 = np.zeros((n2_plus_1, n2_plus_1), dtype=np.int32)
        dummy_adj_1[:-1, :-1] = adj_1
        dummy_adj_2[:-1, :-1] = adj_2

        # Get dummy node features
        dummy_node_feature_1 = np.vstack(
            [self.g1.node_feature, np.zeros((1, self.g1.node_feat_dim), 
            dtype=self.precision)]
        )
        dummy_node_feature_2 = np.vstack(
            [self.g2.node_feature, np.zeros((1, self.g2.node_feat_dim), 
            dtype=self.precision)]
        )

        # Get node costs
        node_mapping = [0, self.node_sub_cost, self.node_ins_cost, self.node_del_cost]
        node_mapping = np.array(node_mapping)
        node_diff = dummy_node_feature_1[:, None, :] - dummy_node_feature_2[None, :, :]
        node_diff: np.ndarray = np.sum(np.abs(node_diff), axis=-1)
        node_diff = (node_diff > self.threshold).astype(np.int32)
        node_diff[-1, :] = 2 # Insert
        node_diff[:, -1] = 3 # Delete
        node_diff[-1, -1] = 0 # self-eps
        node_cost = node_mapping[node_diff]

        # Get edge costs
        edge_mapping = [self.edge_ins_cost, self.edge_sub_cost, self.edge_del_cost]
        edge_mapping = np.array(edge_mapping)
        edge_diff = dummy_adj_1.reshape(-1, 1) - dummy_adj_2.reshape(1, -1) + 1
        edge_diff = edge_diff.astype(np.int32)
        edge_cost = edge_mapping[edge_diff]
        edge_cost = edge_cost.reshape(n1_plus_1, n1_plus_1, n2_plus_1, n2_plus_1)
        edge_cost = edge_cost.transpose(0, 2, 1, 3).reshape(n1_plus_1*n2_plus_1, n1_plus_1*n2_plus_1)
        edge_cost = edge_cost / 2
        
        # Get cost matrix
        K = edge_cost.copy()
        K[np.arange(K.shape[0]), np.arange(K.shape[0])] = node_cost.reshape(-1)
        return K
           
    def from_data(
        self,
        g1: QAPGraphBase = None,
        g2: QAPGraphBase = None,
        sol: np.ndarray = None,
        ref: bool = False,
    ):
        # Set Attributes
        if g1 is not None:
            self.g1 = g1
            n1 = g1.nodes_num
        if g2 is not None:
            self.g2 = g2
            n2 = g2.nodes_num

        # Build Affinity Matrix
        if g1 is not None or g2 is not None:
            K = self.build_cost_mat()
        else:
            K = None

        # Call super ``from_data``
        super().from_data(K=K, n1=n1, n2=n2, sol=sol, ref=ref)