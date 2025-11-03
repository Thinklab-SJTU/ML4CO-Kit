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
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.task.graphset.base import GraphSetTaskBase

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
    
    def binormal_gen(self, size: int, dim: int) -> np.ndarray:
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
            features = self.binormal_gen(size, dim)
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
        nodes_num_scale: tuple = (200, 300),
        nodes_feat_dim_scal: tuple = (1, 10),
        edges_feat_dim_scal: tuple = (1, 10),
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
        
        self.nodes_feat_dim_min, self.nodes_feat_dim_max = nodes_feat_dim_scal
        self.nodes_feat_dim = np.random.randint(self.nodes_feat_dim_min, self.nodes_feat_dim_max+1)
        
        self.edges_feat_dim_min, self.edges_feat_dim_max = edges_feat_dim_scal
        self.edges_feat_dim = np.random.randint(self.edges_feat_dim_min, self.edges_feat_dim_max+1)
        
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
        
        # Single Graph Generation Function 
        self._single_graph_generate= {
            GRAPH_TYPE.BA: self._generate_barabasi_albert_graph,
            GRAPH_TYPE.ER: self._generate_erdos_renyi_graph,
            GRAPH_TYPE.HK: self._generate_holme_kim_graph,
            GRAPH_TYPE.RB: self._generate_rb_graph,
            GRAPH_TYPE.WS: self._generate_watts_strogatz_graph,
        }
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            graph_type: lambda gt=graph_type :self._generate_task(gt)
            for graph_type in [GRAPH_TYPE.ER, GRAPH_TYPE.BA, GRAPH_TYPE.HK, GRAPH_TYPE.WS, GRAPH_TYPE.RB]
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

    def _isomorphic_graph_generate(self, nx_graph: nx.Graph) -> tuple[nx.Graph, np.ndarray]:
        # Build assignment matrix
        num_nodes = len(nx_graph.nodes)
        X_gt = np.zeros((num_nodes, num_nodes))
        permutation =  np.random.permutation(num_nodes)
        X_gt[np.arange(0, num_nodes, dtype=np.int32), permutation] = 1
        X_flat = X_gt.ravel()
        
        # Build isomorphic graph
        mapping = {old: new for old, new in enumerate(permutation)}
        new_graph = nx.relabel_nodes(nx_graph, mapping).copy()
        new_graph = nx.convert_node_labels_to_integers(new_graph, ordering="sorted")
      
        return new_graph, X_flat
     
    def _induced_subgraph_generate(self, nx_graph: nx.Graph, keep_ratio: float = 0.5) ->tuple[nx.Graph, np.ndarray]:
        nodes = np.array(nx_graph.nodes())
        n = len(nodes)
        k = max(1, int(n * keep_ratio))
    
        # Random choose subnodes
        sub_nodes = np.random.choice(nodes, size=k, replace=False)
        sub_graph_view = nx_graph.subgraph(sub_nodes)
        sub_graph = nx.Graph()
        for node in sub_nodes:
            sub_graph.add_node(node, **nx_graph.nodes[node])
        for u, v in sub_graph_view.edges():
            sub_graph.add_edge(u, v, **nx_graph.edges[u,v])
        mapping = {old_label: new_label for new_label, old_label in enumerate(sub_nodes)}
        sub_graph = nx.relabel_nodes(sub_graph, mapping)
        sub_graph = nx.convert_node_labels_to_integers(sub_graph, ordering="sorted")
        
        # Build matching matrix 
        match_matrix = np.zeros((n, k), dtype=np.int32)
        old_idx = sub_nodes.astype(int)
        new_idx = np.arange(k, dtype=int)
        match_matrix[old_idx, new_idx] = 1
        
        match_flat = match_matrix.ravel()
        
        return sub_graph, match_flat    
     
    def _perturbed_graph_generate(
        self, 
        nx_graph: nx.Graph,
        add_ratio: float = 0.01,
        remove_ratio: float = 0.01,
        perturb_node_features: bool = False,
        perturb_edge_features: bool = False,
        node_feat_noise_std: float = 0.1,
        edge_feat_noise_std: float = 0.1
        ) -> nx.Graph:
        """Generate a perturbed graph by adding/removing edges and optionally perturbing features."""
        new_graph = nx_graph.copy()
        n = new_graph.number_of_nodes()
        m = new_graph.number_of_edges()

        # Edge perturbation
        edges = list(new_graph.edges())
        num_remove = int(remove_ratio * m)
        if num_remove > 0:
            new_graph.remove_edges_from(random.sample(edges, k=min(num_remove, len(edges))))

        possible_edges = list(itertools.combinations(range(n), 2))
        possible_edges = [e for e in possible_edges if not new_graph.has_edge(*e)]
        num_add = int(add_ratio * len(possible_edges))
        if num_add > 0:
            new_edges = random.sample(possible_edges, k=min(num_add, len(possible_edges)))
            new_edges_feature = self.edge_feature_gen.generate(len(new_edges), self.edges_feat_dim)
            for i, edge in enumerate(new_edges):
                new_graph.add_edge(edge[0], edge[1], feature=new_edges_feature[i])

        # Node feature perturbation if needed
        if perturb_node_features:
            for _, data in new_graph.nodes(data=True):
                if "feature" in data:
                    noise = np.random.normal(0, node_feat_noise_std, size=(self.nodes_feat_dim,)).astype(self.precision)
                    data["feature"] = data["feature"] + noise
        
        # Edge feature perturbation if needed
        if perturb_edge_features:
            for _, data in new_graph.edges(data=True):
                if "feature" in data:
                    noise = np.random.normal(0, edge_feat_noise_std, size=(self.edges_feat_dim,)).astype(self.precision)
                    data["feature"] = data["feature"] + noise

        return new_graph     
        
    def _generate_feature(self, nx_graph: nx.Graph) -> nx.Graph:
        """Assign feature to nodes and edges."""
        # Add feature to nodes if specified
        nodes_feature = self.node_feature_gen.generate(nx_graph.number_of_nodes(), self.nodes_feat_dim)
        for i, node in enumerate(nx_graph.nodes):
            nx_graph.nodes[node]['feature'] = nodes_feature[i]
        
        # Add feature to edges if specified
        edges_feature = self.edge_feature_gen.generate(nx_graph.number_of_edges(), self.edges_feat_dim)
        for i, edge in enumerate(nx_graph.edges):
            nx_graph.edges[edge]['feature'] = edges_feature[i]
        return nx_graph
    
    def _create_instance(self, nx_graphs: list[nx.Graph]) -> GraphSetTaskBase:
        """Create instance from list of nx.Graph."""
        raise NotImplementedError(
            "Subclasses of GraphGeneratorBase must implement this method."
        )
  
    def _generate_task(self, graph_type: GRAPH_TYPE) -> GraphSetTaskBase:
        """Create task by graph_type."""
        raise NotImplementedError(
            "Subclasses of GraphSetGeneratorBase must implement this method."
        )
    