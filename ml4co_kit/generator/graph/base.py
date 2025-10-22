r"""
Base classes for all graph problem generators.
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
from ml4co_kit.task.graph.base import GraphTaskBase


class GRAPH_TYPE(str, Enum):
    """Define the graph types as an enumeration."""
    ER = "er" # Erdos-Renyi Graph
    BA = "ba" # Barabasi-Albert Graph
    HK = "hk" # Holme-Kim Graph
    WS = "ws" # Watts-Strogatz Graph
    RB = "rb" # RB Graph


class GRAPH_WEIGHT_TYPE(str, Enum):
    """Define the weight types as an enumeration."""
    
    UNIFORM = "uniform" # Uniform Weight
    GAUSSIAN = "gaussian" # Gaussian Weight
    POISSON = "poisson" # Poisson Weight
    EXPONENTIAL = "exponential" # Exponential Weight
    LOGNORMAL = "lognormal" # Lognormal Weight
    POWERLAW = "powerlaw" # Powerlaw Weight
    BINORMIAL = "binomial" # Binomial Weight


class GraphWeightGenerator(object):
    def __init__(
        self,
        weighted_type: GRAPH_WEIGHT_TYPE,
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
        self.weighted_type = weighted_type
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
    
    def uniform_gen(self, size: int) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, size=(size,))
    
    def gaussian_gen(self, size: int) -> np.ndarray:
        return np.random.normal(
            loc=self.gaussian_mean,
            scale=self.gaussian_std,
            size=(size,)
        )
        
    def poisson_gen(self, size: int) -> np.ndarray:
        return np.random.poisson(
            lam=self.poisson_lambda,
            size=(size,)
        )
        
    def exponential_gen(self, size: int) -> np.ndarray:
        return np.random.exponential(
            scale=self.exponential_scale,
            size=(size,)
        )
        
    def lognormal_gen(self, size: int) -> np.ndarray:
        return np.random.lognormal(
            mean=self.lognormal_mean,
            sigma=self.lognormal_sigma,
            size=(size,)
        )
        
    def powerlaw_gen(self, size: int) -> np.ndarray:
        weights = (np.random.pareto(a=self.powerlaw_a, size=(size,)) + 1) * self.powerlaw_b
        noise = np.random.normal(loc=0.0, scale=self.powerlaw_sigma, size=(size,))
        weights += noise
        return weights
    
    def binormal_gen(self, size: int) -> np.ndarray:
        return np.random.binomial(
            n=self.binomial_n,
            p=self.binomial_p,
            size=(size,)
        )
    
    def generate(self, size: int) -> np.ndarray:
        # Generate weights based on the specified type
        if self.weighted_type == GRAPH_WEIGHT_TYPE.UNIFORM:
            weights = self.uniform_gen(size)
        elif self.weighted_type == GRAPH_WEIGHT_TYPE.GAUSSIAN:
            weights = self.gaussian_gen(size)
        elif self.weighted_type == GRAPH_WEIGHT_TYPE.POISSON:
            weights = self.poisson_gen(size)
        elif self.weighted_type == GRAPH_WEIGHT_TYPE.EXPONENTIAL:
            weights = self.exponential_gen(size)
        elif self.weighted_type == GRAPH_WEIGHT_TYPE.LOGNORMAL:
            weights = self.lognormal_gen(size)
        elif self.weighted_type == GRAPH_WEIGHT_TYPE.POWERLAW:
            weights = self.powerlaw_gen(size)
        elif self.weighted_type == GRAPH_WEIGHT_TYPE.BINORMIAL:
            weights = self.binormal_gen(size)
        else:
            raise NotImplementedError(
                f"The weight type {self.weighted_type} is not supported."
            )
        return weights.astype(self.precision)


class GraphGeneratorBase(GeneratorBase):
    """Base class for all graph problem generators."""
    
    def __init__(
        self, 
        task_type: TASK_TYPE, 
        distribution_type: GRAPH_TYPE = GRAPH_TYPE.ER,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num_scale: tuple = (200, 300),
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
        # special args for weighted graph (node/edge weights)
        node_weighted: bool = False,
        node_weighted_gen: GraphWeightGenerator = GraphWeightGenerator(
            weighted_type=GRAPH_WEIGHT_TYPE.UNIFORM),
        edge_weighted: bool = False,
        edge_weighted_gen: GraphWeightGenerator = GraphWeightGenerator(
            weighted_type=GRAPH_WEIGHT_TYPE.UNIFORM),
    ):
        # Super Initialization
        super(GraphGeneratorBase, self).__init__(
            task_type=task_type,
            distribution_type=distribution_type,
            precision=precision
        )

        # Initialize Attributes
        self.nodes_num_min, self.nodes_num_max = nodes_num_scale
        self.nodes_num = np.random.randint(self.nodes_num_min, self.nodes_num_max+1)
        
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
        
        # Special args for weighted graph (node/edge weights)
        self.node_weighted = node_weighted
        self.edge_weighted = edge_weighted
        self.node_weighted_gen = node_weighted_gen
        self.edge_weighted_gen = edge_weighted_gen
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            GRAPH_TYPE.BA: self._generate_barabasi_albert,
            GRAPH_TYPE.ER: self._generate_erdos_renyi,
            GRAPH_TYPE.HK: self._generate_holme_kim,
            GRAPH_TYPE.RB: self._generate_rb,
            GRAPH_TYPE.WS: self._generate_watts_strogatz,
        }
    
    def _generate_barabasi_albert(self) -> GraphTaskBase:
        # Generate Barabasi-Albert graph
        nx_graph: nx.Graph = nx.barabasi_albert_graph(
            n=self.nodes_num, m=min(self.ba_conn_degree, self.nodes_num)
        )
        
        # Add weights to nodes and edges if required
        nx_graph = self._if_need_weighted(nx_graph)
        
        # Create instance from nx.Graph
        return self._create_instance(nx_graph)

    def _generate_erdos_renyi(self) -> GraphTaskBase:
        # Generate Erdos-Renyi graph
        nx_graph: nx.Graph = nx.erdos_renyi_graph(self.nodes_num, self.er_prob)
        
        # Add weights to nodes and edges if required
        nx_graph = self._if_need_weighted(nx_graph)
        
        # Create instance from nx.Graph
        return self._create_instance(nx_graph)
    
    def _generate_holme_kim(self) -> GraphTaskBase:
        # Generate Holme-Kim graph
        nx_graph: nx.Graph = nx.powerlaw_cluster_graph(
            n=self.nodes_num, 
            m=min(self.hk_conn_degree, self.nodes_num), 
            p=self.hk_prob
        )
        
        # Add weights to nodes and edges if required
        nx_graph = self._if_need_weighted(nx_graph)
        
        # Create instance from nx.Graph
        return self._create_instance(nx_graph)
    
    def _generate_watts_strogatz(self) -> GraphTaskBase:
        # Generate Watts-Strogatz graph
        nx_graph: nx.Graph = nx.watts_strogatz_graph(
            n=self.nodes_num, k=self.ws_ring_neighbors, p=self.ws_prob
        )

        # Add weights to nodes and edges if required
        nx_graph = self._if_need_weighted(nx_graph)
        
        # Create instance from nx.Graph
        return self._create_instance(nx_graph)
    
    def _generate_rb(self) -> GraphTaskBase:
        # Get params for RB model (n, k, a)
        while True:
            rb_n = np.random.randint(self.rb_n_min, self.rb_n_max)
            rb_k = np.random.randint(self.rb_k_min, self.rb_k_max)
            rb_v = rb_n * rb_k
            if self.nodes_num_min <= rb_v and self.nodes_num_max >= rb_v:
                break
        self.nodes_num = rb_v
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

        # Add weights to nodes and edges if required
        nx_graph = self._if_need_weighted(nx_graph)
        
        # Create instance from nx.Graph
        return self._create_instance(nx_graph)

    def _if_need_weighted(self, nx_graph: nx.Graph) -> nx.Graph:
        """Assign weights to nodes and/or edges if required."""
        # Add weights to nodes if specified
        if self.node_weighted:
            node_weights = self.node_weighted_gen.generate(self.nodes_num)
            for i, node in enumerate(nx_graph.nodes):
                nx_graph.nodes[node]['weight'] = node_weights[i]
        
        # Add weights to edges if specified
        if self.edge_weighted:
            edge_weights = self.edge_weighted_gen.generate(nx_graph.number_of_edges())
            for i, edge in enumerate(nx_graph.edges):
                nx_graph.edges[edge]['weight'] = edge_weights[i]
        return nx_graph
    
    def _create_instance(self, nx_graph: nx.Graph) -> GraphTaskBase:
        """Create instance from nx.Graph."""
        raise NotImplementedError(
            "Subclasses of GraphGeneratorBase must implement this method."
        )