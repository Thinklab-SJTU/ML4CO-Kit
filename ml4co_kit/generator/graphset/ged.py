r""""
Generator for Graph Edit Distance(GED) instances
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
from ml4co_kit.task.graphset.ged import GEDTask
from ml4co_kit.generator.graphset.base import GRAPH_TYPE
from ml4co_kit.generator.graphset.base import (
    GraphSetGeneratorBase, GRAPH_FEATURE_TYPE, GraphFeatureGenerator
    )

class GRAPH_GENERATE_RULE(str, Enum):
    ISOMORPHIC = "isomorphic"         # Isomorphic Graph
    INDUCED_SUBGRAPH = "induced_subgraph"  # Induced Subgraph
    PERTURBED = "perturbed"           # Perturbed Graph

class GEDGenerator(GraphSetGeneratorBase):
    def __init__(
        self,
        distribution_type: GRAPH_TYPE = GRAPH_TYPE.ER,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num_scale: tuple = (5, 10),
        nodes_feat_dim_scal: tuple = (1, 10),
        edges_feat_dim_scal: tuple = (1, 10),
        graph_generate_rule: GRAPH_GENERATE_RULE =  GRAPH_GENERATE_RULE.ISOMORPHIC,
        # special args for different graph edit problems
        keep_ratio: np.ndarray = 0.5, 
        add_ratio: float = 0.01,
        remove_ratio: float = 0.01,
        perturb_node_features: bool = False,
        perturb_edge_features: bool = False,
        node_feat_noise_std: float = 0.1,
        edge_feat_noise_std: float = 0.1,
        # special args for different distributions (structural)
        er_prob: float = 0.5,
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
        # special args for constructing cost matrix(node/edge)
        node_subst_cost = None,
        node_ins_cost = None,
        node_del_cost = None,
        edge_subst_cost = None,
        edge_ins_cost = None,
        edge_del_cost = None,
    ):
        # Super Initialization
        super(GEDGenerator, self).__init__(
            task_type=TASK_TYPE.GED, 
            distribution_type=distribution_type, 
            precision=precision,
            nodes_num_scale=nodes_num_scale,
            nodes_feat_dim_scal=nodes_feat_dim_scal,
            edges_feat_dim_scal=edges_feat_dim_scal,
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
        
        self.node_subst_cost = node_subst_cost
        self.node_ins_cost = node_ins_cost
        self.node_del_cost = node_del_cost
        self.edge_subst_cost = edge_subst_cost
        self.edge_ins_cost = edge_ins_cost
        self.edge_del_cost = edge_del_cost
    
    def _create_instance(
        self,
        nx_graphs: list[nx.Graph],
        sol: np.ndarray = None,
        ref:bool = False,
        ) -> GEDTask:
        # Check num of graphs
        if len(nx_graphs) != 2:
           raise ValueError("There must be two graphs")
        
        data = GEDTask(precision=self.precision)
        graphs=[]
        for nx_graph in nx_graphs:
            graph = Graph(precision=self.precision)
            graph.from_networkx(nx_graph)
            graphs.append(graph)
        data.from_data(graphs, sol, ref)
        data._deal_with_self_loop()
        
        graph1 = data.graphs[0]
        graph2 = data.graphs[1]
        
        node_feat1 = graph1.nodes_feature
        node_feat2 = graph2.nodes_feature
        edge_feat1 = graph1.edges_feature
        edge_feat2 = graph2.edges_feature
        
        con1 = graph1.edge_index.T
        con2 = graph2.edge_index.T
        
        n1 = graph1.nodes_num
        n2 = graph2.nodes_num
        
        ne1 = graph1.edges_num 
        ne2 = graph2.edges_num
        
        data.cost_mat = data.build_cost_mat(
            node_feat1=node_feat1, 
            edge_feat1=edge_feat1,
            connectivity1=con1,
            node_feat2=node_feat2, 
            edge_feat2=edge_feat2,
            connectivity2=con2,
            n1=n1,
            ne1=ne1,
            n2=n2,
            ne2=ne2,
            node_subst_cost=self.node_subst_cost,
            node_ins_cost=self.node_ins_cost,
            node_del_cost=self.node_del_cost,
            edge_subst_cost=self.edge_subst_cost,
            edge_ins_cost=self.edge_ins_cost,
            edge_del_cost=self.edge_del_cost,
            )
        
        return data
        
    def _generate_task(self) -> GEDTask:
        nx_graph_base: nx.Graph = self._single_graph_generate[self.distribution_type]()
        graph_base = Graph(precision=self.precision)
        graph_base.from_networkx(nx_graph_base)
        
        # Generate a new graph with reference solution by rule
        ref_sol: np.ndarray = None
        if self.graph_generate_rule == GRAPH_GENERATE_RULE.ISOMORPHIC:
            graph_gened, ref_sol = self._isomorphic_graph_generate(graph_base)
            ref_sol = np.pad(ref_sol,((0, 1), (0, 1)), mode = 'constant')
            
        elif self.graph_generate_rule == GRAPH_GENERATE_RULE.INDUCED_SUBGRAPH:
            graph_gened, ref_sol =self._induced_subgraph_generate(graph_base, keep_ratio=self.keep_ratio)
            unmatched = np.where(ref_sol.sum(axis=1) == 0)[0]
            ref_sol = np.pad(ref_sol,((0, 1), (0, 1)), mode = 'constant')
            ref_sol[unmatched, -1] = 1
            
        elif self.graph_generate_rule == GRAPH_GENERATE_RULE.PERTURBED:
            graph_gened, ref_sol = self._perturbed_graph_generate(graph_base)
            ref_sol = np.pad(ref_sol,((0, 1), (0, 1)), mode = 'constant')
        else:
            raise ValueError("This generate rule is not supported for GED.")
        
        data = GEDTask(precision=self.precision)
        data.from_data([graph_base, graph_gened], ref_sol, True)
        data._deal_with_self_loop()
        
        data.cost_mat = data.build_cost_mat()
        
        return data