

r"""
Graph Matching (GM) Generator.
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


import numpy as np
import networkx as nx
from enum import Enum
from typing import Union
from ml4co_kit.task.qap.gm import GMTask
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.base import QAPGraphBase
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.generator.qap.base import QAPGraphGenerator


class GM_TYPE(str, Enum):
    """Define the graph types as an enumeration."""
    ISO = "iso" # Isomorphic Graph
    SUB = "sub" # Subgraph


class GMGenerator(GeneratorBase):
    """Generate GM Tasks."""
    
    def __init__(
        self, 
        distribution_type: GM_TYPE = GM_TYPE.ISO,
        qap_graph_generator: QAPGraphGenerator = QAPGraphGenerator(),
        noise_std: float = 0.1,
        sub_ratio_scale: tuple = (0.3, 0.7),
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        # Super Initialization
        super(GMGenerator, self).__init__(
            task_type=TASK_TYPE.GM,
            distribution_type=distribution_type,
            precision=precision
        )

        # Initialize Attributes
        self.qap_graph_generator = qap_graph_generator
        self.sub_ratio_min, self.sub_ratio_max = sub_ratio_scale
        self.noise_std = noise_std

        # Generation Function Dictionary
        self.generate_func_dict = {
            GM_TYPE.ISO: self._generate_iso,
            GM_TYPE.SUB: self._generate_sub,
        }
    
    def _generate_iso(self):
        # Generate a graph
        g1 = self.qap_graph_generator.generate()
        n1 = g1.nodes_num

        # Randomly generate node permutation
        perm = np.random.permutation(n1)

        # Generate ground truth permutation matrix
        x_gt = np.zeros((n1, n1), dtype=self.precision)
        x_gt[np.arange(n1, dtype=np.int32), perm] = 1
        
        # Transform node features (if exists)
        # perm[old_id] = new_id, so node_feature[old_id] -> node_feature_2[new_id]
        # i.e., node_feature_2[perm] = node_feature
        if g1.node_feature is not None:
            node_feature_2 = np.zeros_like(g1.node_feature)
            node_feature_2[perm] = g1.node_feature
            small_noise = np.random.normal(
                scale=self.noise_std, size=node_feature_2.shape
            ).astype(self.precision)
            node_feature_2 = node_feature_2 + small_noise
        else:
            node_feature_2 = None
        
        # Transform edge_index according to permutation
        # For each edge (u, v), it becomes (perm[u], perm[v])
        edge_index_2 = perm[g1.edge_index]
        
        # Edge features remain the same (same edge order after node permutation)
        if g1.edge_feature is not None:
            edge_feature_2 = np.zeros_like(g1.edge_feature)
            edge_feature_2[perm] = g1.edge_feature
            small_noise = np.random.normal(
                scale=self.noise_std, size=edge_feature_2.shape
            ).astype(self.precision)
            edge_feature_2 = edge_feature_2 + small_noise
        else:
            edge_feature_2 = None
        
        # Create second graph
        g2 = QAPGraphBase(precision=self.precision)
        g2.from_data(
            node_feature=node_feature_2,
            edge_feature=edge_feature_2,
            edge_index=edge_index_2
        )
        
        # Create GM task
        gm_task = GMTask(precision=self.precision)
        gm_task.from_data(g1=g1, g2=g2, sol=x_gt, ref=True)
        
        return gm_task

    def _generate_sub(self):
        """
        Generate subgraph matching task where g1 is a subgraph of g2.
        g1 is selected as a spatially clustered subset of g2's nodes.
        """
        # Generate the larger graph g2
        g2 = self.qap_graph_generator.generate()
        n2 = g2.nodes_num
        
        # Determine size of subgraph g1 (smaller than g2)
        n1 = int(n2 * np.random.uniform(self.sub_ratio_min, self.sub_ratio_max))
        n1 = max(3, n1)
        
        # Get adjacency matrix for spatial layout
        A2 = g2.to_adj_matrix()
        
        # Use spring layout to get spatial positions for node selection
        G2_nx = nx.from_numpy_array(A2)
        pos2 = nx.spring_layout(G2_nx)
        pos2_array = np.array([pos2[i] for i in range(n2)])
        
        # Select nodes for g1 as a spatially clustered subset
        selected = [0]  # start with node 0
        unselected = list(range(1, n2))
        
        while len(selected) < n1:
            # Calculate distance from selected nodes to unselected nodes
            selected_pos = pos2_array[selected]  # (len(selected), 2)
            unselected_pos = pos2_array[unselected]  # (len(unselected), 2)
            
            # Compute pairwise distances and sum over selected nodes
            dist = np.sum(
                np.sum(
                    np.abs(
                        np.expand_dims(selected_pos, 1) - np.expand_dims(unselected_pos, 0)
                    ), 
                    axis=-1
                ), 
                axis=0
            )
            
            # Select the closest unselected node
            select_idx = np.argmin(dist).item()
            select_id = unselected[select_idx]
            selected.append(select_id)
            unselected.remove(select_id)
        
        selected.sort()
        
        # Extract subgraph g1 from g2
        # Node features
        if g2.node_feature is not None:
            node_feature_1 = g2.node_feature[selected]
            small_noise = np.random.normal(
                scale=self.noise_std, size=node_feature_1.shape
            ).astype(self.precision)
            node_feature_1 = node_feature_1 + small_noise
        else:
            node_feature_1 = None
        
        # Edge index and edge features
        # Build mapping: old_node_id -> new_node_id in g1
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(selected)}
        
        # Filter edges that have both endpoints in selected nodes
        edge_mask = np.isin(g2.edge_index[0], selected) & np.isin(g2.edge_index[1], selected)
        edge_index_1 = g2.edge_index[:, edge_mask]
        
        # Remap node IDs in edge_index
        edge_index_1_remapped = np.array([
            [node_mapping[edge_index_1[0, i]], node_mapping[edge_index_1[1, i]]]
            for i in range(edge_index_1.shape[1])
        ]).T
        
        # Extract corresponding edge features
        if g2.edge_feature is not None:
            edge_feature_1 = g2.edge_feature[edge_mask]
            small_noise = np.random.normal(
                scale=self.noise_std, size=edge_feature_1.shape
            ).astype(self.precision)
            edge_feature_1 = edge_feature_1 + small_noise
        else:
            edge_feature_1 = None
        
        # Create g1
        g1 = QAPGraphBase(precision=self.precision)
        g1.from_data(
            node_feature=node_feature_1,
            edge_feature=edge_feature_1,
            edge_index=edge_index_1_remapped
        )
        
        # Create ground truth assignment matrix X_gt
        # X_gt[i, j] = 1 if node i in g1 corresponds to node j in g2
        X_gt = np.zeros((n1, n2), dtype=self.precision)
        for i, j in enumerate(selected):
            X_gt[i, j] = 1
        
        # Create GM task
        gm_task = GMTask(precision=self.precision)
        gm_task.from_data(g1=g1, g2=g2, sol=X_gt, ref=True)
        
        return gm_task