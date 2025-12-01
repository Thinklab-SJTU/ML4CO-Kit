r""""
Generator for Graph Matching(GM) instances
"""

# Copyright (c) 2024 Thinklab@SJTUS
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
import networkx as nx
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.graphset.base import Graph
from ml4co_kit.task.graphset.gm import GMTask
from ml4co_kit.generator.graphset.base import GRAPH_TYPE
from ml4co_kit.generator.graphset.base import (
    GraphSetGeneratorBase, GRAPH_FEATURE_TYPE, GraphFeatureGenerator
    )

class GRAPH_GENERATE_RULE(str, Enum):
    ISOMORPHIC = "isomorphic"         # Isomorphic Graph
    INDUCED_SUBGRAPH = "induced_subgraph"  # Induced Subgraph
    PERTURBED = "perturbed"           # Perturbed Graph

class GMGenerator(GraphSetGeneratorBase):
    def __init__(
        self,
        distribution_type: GRAPH_TYPE = GRAPH_TYPE.ER,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num_scale: tuple = (50, 100),
        node_feat_dim_scale: tuple = (1, 10),
        edge_feat_dim_scale: tuple = (1, 10),
        graph_generate_rule: GRAPH_GENERATE_RULE =  GRAPH_GENERATE_RULE.ISOMORPHIC,
        # special args for different graph matching problrms
        keep_ratio: np.ndarray = 0.5, 
        add_ratio: float = 0.01,
        remove_ratio: float = 0.01,
        perturb_node_features: bool = False,
        perturb_edge_features: bool = False,
        node_feat_noise_std: float = 0.1,
        edge_feat_noise_std: float = 0.1,
        # special args for different distributions (structural)
        er_prob: float = 0.15,
        ba_conn_degree: int = 4,
        hk_prob: float = 0.3,
        hk_conn_degree: int = 10,
        ws_prob: float = 0.3,
        ws_ring_neighbors: int = 2,
        rb_n_scale: tuple = (20, 25),
        rb_k_scale: tuple = (5, 12),
        rb_p_scale: tuple = (0.3, 1.0),
        # special args for featured graph
        node_feature_gen: GraphFeatureGenerator = GraphFeatureGenerator(
            feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
        edge_feature_gen: GraphFeatureGenerator = GraphFeatureGenerator(
            feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
        # special args for constructing affinity matrix(node/edge)
        node_aff_fn = None,
        edge_aff_fn = None
    ):
        # Super Initialization
        super(GMGenerator, self).__init__(
            task_type=TASK_TYPE.GM, 
            distribution_type=distribution_type, 
            precision=precision,
            nodes_num_scale=nodes_num_scale,
            node_feat_dim_scale=node_feat_dim_scale,
            edge_feat_dim_scale=edge_feat_dim_scale,
            er_prob=er_prob,
            ba_conn_degree=ba_conn_degree,
            hk_prob=hk_prob,
            hk_conn_degree=hk_conn_degree,
            ws_prob=ws_prob,
            ws_ring_neighbors=ws_ring_neighbors,
            rb_n_scale=rb_n_scale,
            rb_k_scale=rb_k_scale,
            rb_p_scale=rb_p_scale,
            node_feature_gen=node_feature_gen,
            edge_feature_gen=edge_feature_gen
        )
        
        self.keep_ratio = keep_ratio
        self.add_ratio = add_ratio
        self.remove_ratio = remove_ratio
        self.perturb_node_features = perturb_node_features
        self.perturb_edge_features = perturb_edge_features
        self.node_feat_noise_std = node_feat_noise_std
        self.edge_feat_noise_std = edge_feat_noise_std
        
        # GM task defined by graph generate rule
        self.graph_generate_rule=graph_generate_rule 
        
        self.node_aff_fn = node_aff_fn
        self.edge_aff_fn = edge_aff_fn
      
    def _generate_task(self) -> GMTask:
        nx_graph_base: nx.Graph = self._generate_single_graph[self.distribution_type]()
        
        # check edges of the generated graph
        if nx_graph_base.number_of_edges() == 0:
            raise ValueError("Generated base graph has no edges, please adjust the graph generation parameters.")
        
        graph_base = Graph(precision=self.precision)
        graph_base.from_networkx(nx_graph_base)
        # Generate a new graph with reference solution by rule
        ref_sol: np.ndarray = None
        if self.graph_generate_rule == GRAPH_GENERATE_RULE.ISOMORPHIC:
            graph_gened, ref_sol = self._isomorphic_graph_generate(graph_base)
        elif self.graph_generate_rule == GRAPH_GENERATE_RULE.INDUCED_SUBGRAPH:
            graph_gened, ref_sol =self._induced_subgraph_generate(graph_base, keep_ratio=self.keep_ratio)
        elif self.graph_generate_rule == GRAPH_GENERATE_RULE.PERTURBED:
            graph_gened, ref_sol = self._perturbed_graph_generate(
                graph_base, 
                add_ratio=self.add_ratio,
                remove_ratio=self.remove_ratio,
                perturb_node_features=self.perturb_node_features,
                perturb_edge_features=self.perturb_edge_features,
                node_feat_noise_std=self.node_feat_noise_std,
                edge_feat_noise_std=self.edge_feat_noise_std
            )
        else:
            raise ValueError("This generate rule is not supported for GM.")
        
        # check edges of the generated graph
        if graph_gened.edges_num == 0:
            raise ValueError("Generated graph has no edges, please adjust the graph generation parameters.")
        
        data = GMTask(precision=self.precision)
        data.from_data([graph_base, graph_gened], ref_sol, True)
        data._deal_with_self_loop()
        
        data.aff_mat = data.build_aff_mat(
            node_aff_fn=self.node_aff_fn,
            edge_aff_fn=self.edge_aff_fn
            )
        
        return data
        
        