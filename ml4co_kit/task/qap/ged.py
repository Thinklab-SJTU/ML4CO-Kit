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
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(GEDTask, self).__init__(
            task_type=TASK_TYPE.GED,
            minimize=True,
            precision=precision
        )
        
        # Initialize Attributes (Cost)
        self.node_sub_cost = node_sub_cost
        self.node_ins_cost = node_ins_cost
        self.node_del_cost = node_del_cost

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
        n12_plus_1 = max(n1_plus_1, n2_plus_1)
        adj_1 = self.g1.to_adj_matrix()
        adj_2 = self.g2.to_adj_matrix()

        # Get dummy adjacency matrices
        dummy_adj_1 = np.zeros((n12_plus_1, n12_plus_1), dtype=np.int32)
        dummy_adj_2 = np.zeros((n12_plus_1, n12_plus_1), dtype=np.int32)
        dummy_adj_1[0:n1, 0:n1] = adj_1
        dummy_adj_2[0:n2, 0:n2] = adj_2

        # Get dummy node features
        dummy_node_feature_1 = np.zeros(
            (n12_plus_1, self.g1.node_feat_dim), 
            dtype=self.precision
        )
        dummy_node_feature_2 = np.zeros(
            (n12_plus_1, self.g2.node_feat_dim), 
            dtype=self.precision
        )
        dummy_node_feature_1[0:n1, :] = self.g1.node_feature
        dummy_node_feature_2[0:n2, :] = self.g2.node_feature

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

        # Get mask
        mask_1 = np.ones_like(dummy_adj_1)
        mask_2 = np.ones_like(dummy_adj_2)
        np.fill_diagonal(mask_1, 0)
        np.fill_diagonal(mask_2, 0)

        # Get edge costs
        dummy_adj_1 = dummy_adj_1.reshape(-1, 1)
        dummy_adj_2 = dummy_adj_2.reshape(1, -1)
        mask_1 = mask_1.reshape(-1, 1)
        mask_2 = mask_2.reshape(1, -1)
        k: np.ndarray = np.abs(dummy_adj_1 - dummy_adj_2) * np.matmul(mask_1, mask_2)
        k[np.logical_not(np.matmul(mask_1, mask_2))] = 1e5
        k = k.reshape(n12_plus_1, n12_plus_1, n12_plus_1, n12_plus_1)
        k = k.transpose([0, 2, 1, 3])
        k = k.reshape(n12_plus_1 * n12_plus_1, n12_plus_1 * n12_plus_1)
        k = k / 2

        # Get cost matrix
        K = k.copy()
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
            n1 = g1.nodes_num + 1
        else:
            n1 = None
        if g2 is not None:
            self.g2 = g2
            n2 = g2.nodes_num + 1
        else:
            n2 = None

        if n1 is not None:
            max_n1_n2 = max(n1, n2)
        else:
            max_n1_n2 = None

        # Build Affinity Matrix
        if g1 is not None or g2 is not None:
            K = self.build_cost_mat()
        else:
            K = None

        # Call super ``from_data``
        super().from_data(
            K=K, n1=max_n1_n2, n2=max_n1_n2, sol=sol, ref=ref
        )