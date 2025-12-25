r"""
Base classes for all graph set problem generators.
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


import random
import itertools
import numpy as np
import networkx as nx
from enum import Enum
from typing import Union, Tuple
from ml4co_kit.task.base import TASK_TYPE, TaskBase
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.task.graphset.base import Graph, GraphSetTaskBase


class GRAPH_TYPE(str, Enum):
    """Define the graph types as an enumeration."""
    ER = "er" # Erdos-Renyi Graph
    BA = "ba" # Barabasi-Albert Graph
    HK = "hk" # Holme-Kim Graph
    WS = "ws" # Watts-Strogatz Graph
    RB = "rb" # RB Graph


class GRAPH_FEATURE_TYPE(str, Enum):
    """Define the featrue types as an enumeration."""
    
    UNIFORM = "uniform" # Uniform Feature
    GAUSSIAN = "gaussian" # Gaussian Feature
    POISSON = "poisson" # Poisson Feature
    EXPONENTIAL = "exponential" # Exponential Feature
    LOGNORMAL = "lognormal" # Lognormal Feature
    POWERLAW = "powerlaw" # Powerlaw Feature
    BINOMIAL = "binomial" # Binomial Feature

class GraphFeatureGenerator(object):
    def __init__(
        self,
        feature_type: GRAPH_FEATURE_TYPE,
        precision: Union[np.float32, np.float64] = np.float32,
        # gaussian
        gaussian_mean: float = 0.0,
        gaussian_std: float = 1.0,
        # poisson
        poisson_lambda: float = 1.0,
        # exponential
        exponential_scale: float = 1.0,
        # lognormal
        lognormal_mean: float = 0.0,
        lognormal_sigma: float = 1.0,
        # powerlaw
        powerlaw_a: float = 1.0,
        powerlaw_b: float = 10.0,
        powerlaw_sigma: float = 1.0,
        # binomial
        binomial_n: int = 10,
        binomial_p: float = 0.5,
    ) -> None:
        # Initialize Attributes
        self.feature_type = feature_type
        self.precision = precision
        
        # Special Args for Gaussian
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        
        # Special Args for Poisson
        self.poisson_lambda = poisson_lambda
        
        # Special Args for Exponential
        self.exponential_scale = exponential_scale
        
        # Special Args for Lognormal
        self.lognormal_mean = lognormal_mean
        self.lognormal_sigma = lognormal_sigma
        
        # Special Args for Powerlaw
        self.powerlaw_a = powerlaw_a
        self.powerlaw_b = powerlaw_b
        self.powerlaw_sigma = powerlaw_sigma
        
        # Special Args for Binomial
        self.binomial_n = binomial_n
        self.binomial_p = binomial_p
    
    def uniform_gen(self, size: int, dim: int) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, size=(size, dim))
    
    def gaussian_gen(self, size: int, dim: int) -> np.ndarray:
        return np.random.normal(
            loc=self.gaussian_mean,
            scale=self.gaussian_std,
            size=(size, dim)
        )
        
    def poisson_gen(self, size: int, dim: int) -> np.ndarray:
        return np.random.poisson(
            lam=self.poisson_lambda,
            size=(size, dim)
        )
        
    def exponential_gen(self, size: int, dim: int) -> np.ndarray:
        return np.random.exponential(
            scale=self.exponential_scale,
            size=(size, dim)
        )
        
    def lognormal_gen(self, size: int, dim: int) -> np.ndarray:
        return np.random.lognormal(
            mean=self.lognormal_mean,
            sigma=self.lognormal_sigma,
            size=(size, dim)
        )
        
    def powerlaw_gen(self, size: int, dim: int) -> np.ndarray:
        features = (np.random.pareto(a=self.powerlaw_a, size=(size, dim)) + 1) * self.powerlaw_b
        noise = np.random.normal(loc=0.0, scale=self.powerlaw_sigma, size=(size, dim))
        features += noise
        return features
    
    def binomal_gen(self, size: int, dim: int) -> np.ndarray:
        return np.random.binomial(
            n=self.binomial_n,
            p=self.binomial_p,
            size=(size, dim)
        )
    
    def generate(self, size: int, dim: int) -> np.ndarray:
        # Generate features based on the specified type
        if self.feature_type == GRAPH_FEATURE_TYPE.UNIFORM:
            features = self.uniform_gen(size, dim)
        elif self.feature_type == GRAPH_FEATURE_TYPE.GAUSSIAN:
            features = self.gaussian_gen(size, dim)
        elif self.feature_type == GRAPH_FEATURE_TYPE.POISSON:
            features = self.poisson_gen(size, dim)
        elif self.feature_type == GRAPH_FEATURE_TYPE.EXPONENTIAL:
            features = self.exponential_gen(size, dim)
        elif self.feature_type == GRAPH_FEATURE_TYPE.LOGNORMAL:
            features = self.lognormal_gen(size, dim)
        elif self.feature_type == GRAPH_FEATURE_TYPE.POWERLAW:
            features = self.powerlaw_gen(size, dim)
        elif self.feature_type == GRAPH_FEATURE_TYPE.BINOMIAL:
            features = self.binomal_gen(size, dim)
        else:
            raise NotImplementedError(
                f"The feature type {self.feature_type} is not supported."
            )
        return features.astype(self.precision)
    
class GraphSetGeneratorBase(GeneratorBase):
    """Base class for all graph set problem generators."""
    
    def __init__(
        self, 
        task_type: TASK_TYPE, 
        distribution_type: GRAPH_TYPE = GRAPH_TYPE.ER,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num_scale: tuple = (10, 20),
        node_feat_dim_scale: tuple = (1, 10),
        edge_feat_dim_scale: tuple = (1, 10),
        # special args for different distributions (structural)
        er_prob: float = 0.15,
        ba_conn_degree: int = 10,
        hk_prob: float = 0.3,
        hk_conn_degree: int = 10,
        ws_prob: float = 0.3,
        ws_ring_neighbors: int = 2,
        rb_n_scale: tuple = (20, 25),
        rb_k_scale: tuple = (5, 12),
        rb_p_scale: tuple = (0.3, 1.0),
        # special args for featured graph (node/edge features)
        node_feature_gen: GraphFeatureGenerator = GraphFeatureGenerator(
            feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
        edge_feature_gen: GraphFeatureGenerator = GraphFeatureGenerator(
            feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
    ):
        # Super Initialization
        super(GraphSetGeneratorBase, self).__init__(
            task_type=task_type,
            distribution_type=distribution_type,
            precision=precision
        )

        # Initialize Attributes
        self.nodes_num_min, self.nodes_num_max = nodes_num_scale
        self.nodes_num_base = np.random.randint(self.nodes_num_min, self.nodes_num_max+1)
        
        self.node_feat_dim_min, self.node_feat_dim_max = node_feat_dim_scale
        self.node_feat_dim = np.random.randint(self.node_feat_dim_min, self.node_feat_dim_max+1)
        
        self.edge_feat_dim_min, self.edge_feat_dim_max = edge_feat_dim_scale
        self.edge_feat_dim = np.random.randint(self.edge_feat_dim_min, self.edge_feat_dim_max+1)
        
        # Special args for different distributions (structural)
        self.er_prob = er_prob
        self.ba_conn_degree = ba_conn_degree
        self.hk_prob = hk_prob
        self.hk_conn_degree = hk_conn_degree
        self.ws_prob = ws_prob
        self.ws_ring_neighbors = ws_ring_neighbors
        self.rb_n_min, self.rb_n_max = rb_n_scale
        self.rb_k_min, self.rb_k_max = rb_k_scale
        self.rb_p_min, self.rb_p_max = rb_p_scale
        
        # Special args for featured graph (node/edge features)
        self.node_feature_gen = node_feature_gen
        self.edge_feature_gen = edge_feature_gen
        
        # Single Graph Generate Function 
        self._generate_single_graph= {
            GRAPH_TYPE.BA: self._generate_barabasi_albert_graph,
            GRAPH_TYPE.ER: self._generate_erdos_renyi_graph,
            GRAPH_TYPE.HK: self._generate_holme_kim_graph,
            GRAPH_TYPE.RB: self._generate_rb_graph,
            GRAPH_TYPE.WS: self._generate_watts_strogatz_graph,
        }
      
    def _generate_barabasi_albert_graph(self):
        # Generate Barabasi-Albert graph
        nx_graph: nx.Graph = nx.barabasi_albert_graph(
            n=self.nodes_num_base, m=min(self.ba_conn_degree, self.nodes_num_base)
        )
        
        # Add features to nodes and edges 
        nx_graph = self._generate_feature(nx_graph)
        
        return nx_graph

    def _generate_erdos_renyi_graph(self):
        # Generate Erdos-Renyi graph
        nx_graph: nx.Graph = nx.erdos_renyi_graph(self.nodes_num_base, self.er_prob)
        
        # Add features to nodes and edges
        nx_graph = self._generate_feature(nx_graph)
        
        return nx_graph
    
    def _generate_holme_kim_graph(self):
        # Generate Holme-Kim graph
        nx_graph: nx.Graph = nx.powerlaw_cluster_graph(
            n=self.nodes_num_base, 
            m=min(self.hk_conn_degree, self.nodes_num_base), 
            p=self.hk_prob
        )
        
        # Add features to nodes and edges
        nx_graph = self._generate_feature(nx_graph)
        
        return nx_graph
    
    def _generate_watts_strogatz_graph(self):
        # Generate Watts-Strogatz graph
        nx_graph: nx.Graph = nx.watts_strogatz_graph(
            n=self.nodes_num_base, k=self.ws_ring_neighbors, p=self.ws_prob
        )

        # Add features to nodes and edges 
        nx_graph = self._generate_feature(nx_graph)
        
        return nx_graph
    
    def _generate_rb_graph(self):
        # Get params for RB model (n, k, a)
        while True:
            rb_n = np.random.randint(self.rb_n_min, self.rb_n_max)
            rb_k = np.random.randint(self.rb_k_min, self.rb_k_max)
            rb_v = rb_n * rb_k
            if self.nodes_num_min <= rb_v and self.nodes_num_max >= rb_v:
                break
        self.nodes_num_base = rb_v
        rb_a = np.log(rb_k) / np.log(rb_n)
        
        # Get params for RB model (p, r, s, iterations)
        rb_p = np.random.uniform(self.rb_p_min, self.rb_p_max)
        rb_r = - rb_a / np.log(1 - rb_p)
        rb_s = int(rb_p * (rb_n ** (2 * rb_a)))
        iterations = int(rb_r * rb_n * np.log(rb_n) - 1)
        
        # Generate RB instance
        parts = np.reshape(np.int64(range(rb_v)), (rb_n, rb_k))
        nand_clauses = []
        for i in parts:
            nand_clauses += itertools.combinations(i, 2)
        edges = set()
        for _ in range(iterations):
            i, j = np.random.choice(rb_n, 2, replace=False)
            all = set(itertools.product(parts[i, :], parts[j, :]))
            all -= edges
            edges |= set(random.sample(tuple(all), k=min(rb_s, len(all))))
        nand_clauses += list(edges)
        clauses = {'NAND': nand_clauses}
        
        # Convert to numpy array
        clauses = {relation: np.int32(clause_list) for relation, clause_list in clauses.items()}
        
        # Convert to nx.Graph
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(clauses['NAND'])

        # Add features to nodes and edges
        nx_graph = self._generate_feature(nx_graph)
        
        return nx_graph

    def _isomorphic_graph_generate(self, graph: Graph) -> Tuple[Graph, np.ndarray]:
        # Build assignment matrix
        nodes_num = graph.nodes_num
        X = np.zeros((nodes_num, nodes_num))
        permutation = np.random.permutation(nodes_num) # graph2 to graph1
        X[np.arange(0, nodes_num, dtype=np.int32), permutation] = 1
        X = X.T
        mapping_1to2 = X.argmax(axis=1)
        
        # Build isomorphic graph
        new_node_feat = graph.node_feature[permutation]
        new_edge_index = mapping_1to2[graph.edge_index]
        iso_graph = Graph(
            nodes_num=nodes_num, 
            node_feature=new_node_feat, 
            edge_index=new_edge_index, 
            edge_feature=graph.edge_feature.copy(),
            node_feat_dim=graph.node_feature.shape[1],
            edge_feat_dim=graph.edge_feature.shape[1]
            )
    
      
        return iso_graph, X
     
    def _induced_subgraph_generate(self, graph: Graph, keep_ratio: float = 0.5) -> Tuple[Graph, np.ndarray]:
        nodes_num = graph.nodes_num
        nodes = np.arange(nodes_num, dtype=np.int32)
        k = max(1, int(nodes_num * keep_ratio))
    
        # Random choose subnodes
        sub_nodes = np.random.choice(nodes, size=k, replace=False)
        mask = np.zeros(nodes_num, dtype=bool)
        mask[sub_nodes] = True
        
        # Build subgraph
        old_to_new_idx = -np.ones(nodes_num, dtype=np.int32)
        old_to_new_idx[sub_nodes] = np.arange(k, dtype=np.int32)
        
        sub_node_feat = graph.node_feature[sub_nodes]
        edge_index = graph.edge_index
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        sub_edge_feat = graph.edge_feature[edge_mask]
        sub_edge_index = old_to_new_idx[edge_index[:, edge_mask]]
        
        sub_graph = Graph(
            nodes_num=k,
            node_feature=sub_node_feat,
            edge_index=sub_edge_index,
            edge_feature=sub_edge_feat,
            node_feat_dim=graph.node_feature.shape[1],
            edge_feat_dim=graph.edge_feature.shape[1]
        )
        
        # Build matching matrix 
        X = np.zeros((nodes_num, k), dtype=np.int32)
        X[sub_nodes, np.arange(k)] = 1
        
        return sub_graph, X
     
    def _perturbed_graph_generate(
        self, 
        graph: Graph,
        add_ratio: float = 0,
        remove_ratio: float = 0,
        perturb_node_features: bool = False,
        perturb_edge_features: bool = False,
        node_feat_noise_std: float = 0.1,
        edge_feat_noise_std: float = 0.1
        ) -> nx.Graph:
        """Generate a perturbed graph by adding/removing edges and optionally perturbing features."""
        n = graph.nodes_num
        edge_index = graph.edge_index.copy()        
        edge_feat = graph.edge_feature.copy()     
        node_feat = graph.node_feature.copy() 
        
        # Remove edges
        u = edge_index[0]
        v = edge_index[1]
        
        can_u = np.minimum(u, v)
        can_v = np.maximum(u, v)
        can_edges = list(zip(can_u, can_v))
        
        unique_edges = list(set(can_edges))
        num_unique = len(unique_edges)
        
        num_remove = int(num_unique * remove_ratio)
        
        if num_remove > 0:  
            remove_edges = set(random.sample(unique_edges, k=min(num_remove, num_unique)))

            remove_mask = np.array([edge in remove_edges for edge in can_edges])
            keep_mask = ~remove_mask

            edge_index = edge_index[:, keep_mask]
            edge_feat = edge_feat[keep_mask]
        
        # Add edges
        u = edge_index[0]
        v = edge_index[1]
        can_u = np.minimum(u, v)
        can_v = np.maximum(u, v)
        existing_edges = set(zip(can_u, can_v))

        possible_edges = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if (i, j) not in existing_edges
        ]
        
        num_add = 0
        if len(possible_edges) > 0 and add_ratio > 0:
            num_add = int(add_ratio * len(possible_edges))

        if num_add > 0:
            new_edges = random.sample(possible_edges, k=min(num_add, len(possible_edges)))
            new_feat = self.edge_feature_gen.generate(len(new_edges), dim=self.edge_feat_dim)
            add_list = []
            for (u, v) in new_edges:
                add_list.append([u, v])
                add_list.append([v, u])

            add_edge_index = np.array(add_list).T  
            add_feat = np.repeat(new_feat, repeats=2, axis=0)
            
            edge_index = np.concatenate([edge_index, add_edge_index], axis=1)
            edge_feat = np.concatenate([edge_feat, add_feat], axis=0)
        
        # Perturb features
        if perturb_node_features:
            noise = np.random.normal(
                0, node_feat_noise_std,
                size=node_feat.shape
            ).astype(node_feat.dtype)
            node_feat = node_feat + noise
            
        if perturb_edge_features:
            noise = np.random.normal(
                0, edge_feat_noise_std,
                size=edge_feat.shape
            ).astype(edge_feat.dtype)
            edge_feat = edge_feat + noise    
        
        pert_graph = Graph(
            nodes_num=n,
            node_feature=node_feat,
            edge_index=edge_index,
            edge_feature=edge_feat,
            node_feat_dim=graph.node_feature.shape[1],
            edge_feat_dim=graph.edge_feature.shape[1]
        )
        
        X = np.eye(n, dtype=np.int32)
        return pert_graph, X
               
    def _generate_feature(self, nx_graph: nx.Graph) -> nx.Graph:
        """Assign feature to nodes and edges."""
        # Add feature to nodes if specified
        node_feat = self.node_feature_gen.generate(nx_graph.number_of_nodes(), self.node_feat_dim)
        for i, node in enumerate(nx_graph.nodes):
            nx_graph.nodes[node]['feature'] = node_feat[i]
        
        # Add feature to edges if specified
        edge_feat = self.edge_feature_gen.generate(nx_graph.number_of_edges(), self.edge_feat_dim)
        for i, edge in enumerate(nx_graph.edges):
            nx_graph.edges[edge]['feature'] = edge_feat[i]
        return nx_graph
    
    def generate(self) -> TaskBase:
        return self._generate_task()
    
    def _generate_task(self) -> GraphSetTaskBase:
        """Create task by graph_type."""
        raise NotImplementedError(
            "Subclasses of GraphSetGeneratorBase must implement this method."
        )
    