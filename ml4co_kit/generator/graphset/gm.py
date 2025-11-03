import numpy as np
import networkx as nx
import itertools
import random 
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.graphset.base import GraphSetTaskBase,Graph
from ml4co_kit.task.graph.gm import GMTask
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
        nodes_num_scale: tuple = (200, 300),
        nodes_feat_dim_scal: tuple = (1, 10),
        edges_feat_dim_scal: tuple = (1, 10),
        graph_generate_rule: GRAPH_GENERATE_RULE =  GRAPH_GENERATE_RULE.ISOMORPHIC,
        # special args for induced graph
        keep_ratio: np.ndarray = 0.5, 
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
        
        # GM task defined by graph generate rule
        self.graph_generate_rule=graph_generate_rule 
        
        self.node_aff_fn = node_aff_fn
        self.edge_aff_fn = edge_aff_fn
    
    def _create_instance(
        self,
        nx_graphs: list[nx.Graph],
        sol: np.ndarray = None,
        ref:bool = False,
        ) -> GMTask:
        # Check num of graphs
        if len(nx_graphs) != 2:
           raise ValueError("There must be two graphs")
        
        data = GMTask(precision=self.precision)
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
        
        data.aff_matrix = data.build_aff_mat(
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
            node_aff_fn=self.node_aff_fn,
            edge_aff_fn=self.edge_aff_fn
            )
        
        return data
        
    def _generate_task(self, graph_type: GRAPH_TYPE) -> GMTask:
        nx_graph_base: nx.Graph = self._single_graph_generate[self.distribution_type]()
        
        # Generate a new graph with reference solution by rule
        ref_sol: np.ndarray = None
        if self.graph_generate_rule == GRAPH_GENERATE_RULE.ISOMORPHIC:
            nx_graph_gened, ref_sol = self._isomorphic_graph_generate(nx_graph_base)
        elif self.graph_generate_rule == GRAPH_GENERATE_RULE.INDUCED_SUBGRAPH:
            nx_graph_gened, ref_sol =self._induced_subgraph_generate(nx_graph_base, keep_ratio=self.keep_ratio)
        else:
            raise ValueError("This generate rule is not supported for GM.")
        
        return self._create_instance(nx_graphs=[nx_graph_base, nx_graph_gened], sol=ref_sol, ref=True)  