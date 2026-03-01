
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
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.base import QAPGraphBase
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.generator.graph.base import GRAPH_TYPE


class QAPGraphGenerator(object):
    """Generate QAP Graphs."""
    
    def __init__(
        self, 
        distribution_type: GRAPH_TYPE = GRAPH_TYPE.ER,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num_scale: tuple = (10, 20),
        # special args for different distributions (structural)
        er_prob: float = 0.5,
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
        node_feat_dim: int = 10,
        edge_weighted: bool = False,
        edge_feat_dim: int = 10,
    ):
        # Initialize Attributes
        self.precision = precision
        self.distribution_type = distribution_type
        self.nodes_num_min, self.nodes_num_max = nodes_num_scale
        
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
        self.node_feat_dim = node_feat_dim
        self.edge_weighted = edge_weighted
        self.edge_feat_dim = edge_feat_dim

        # Generation Function Dictionary
        self.generate_func_dict = {
            GRAPH_TYPE.BA: self._generate_barabasi_albert,
            GRAPH_TYPE.ER: self._generate_erdos_renyi,
            GRAPH_TYPE.HK: self._generate_holme_kim,
            GRAPH_TYPE.RB: self._generate_rb,
            GRAPH_TYPE.WS: self._generate_watts_strogatz,
        }
    
    def _generate_barabasi_albert(self) -> QAPGraphBase:
        """
        @article{
            barabasi1999emergence,
            title={Emergence of scaling in random networks},
            author={Barab{\'a}si, Albert-L{\'a}szl{\'o} and Albert, R{\'e}ka},
            journal={science},
            volume={286},
            number={5439},
            pages={509--512},
            year={1999},
            publisher={American Association for the Advancement of Science}
        }
        """
        # Generate Barabasi-Albert graph
        nx_graph: nx.Graph = nx.barabasi_albert_graph(
            n=self.nodes_num, m=min(self.ba_conn_degree, self.nodes_num)
        )
        return self._if_need_weighted(nx_graph)

    def _generate_erdos_renyi(self) -> QAPGraphBase:
        """
        @article{
            erd6s1960evolution,
            title={On the evolution of random graphs},
            author={Erd6s, Paul and R{\'e}nyi, Alfr{\'e}d},
            journal={Publ. Math. Inst. Hungar. Acad. Sci},
            volume={5},
            pages={17--61},
            year={1960}
        }
        """
        # Generate Erdos-Renyi graph
        nx_graph: nx.Graph = nx.erdos_renyi_graph(self.nodes_num, self.er_prob)
        return self._if_need_weighted(nx_graph)
    
    def _generate_holme_kim(self) -> QAPGraphBase:
        """
        @article{
            holme2002growing,
            title={Growing scale-free networks with tunable clustering},
            author={Holme, Petter and Kim, Beom Jun},
            journal={Physical review E},
            volume={65},
            number={2},
            pages={026107},
            year={2002},
            publisher={APS}
        }
        """
        # Generate Holme-Kim graph
        nx_graph: nx.Graph = nx.powerlaw_cluster_graph(
            n=self.nodes_num, 
            m=min(self.hk_conn_degree, self.nodes_num), 
            p=self.hk_prob
        )
        return self._if_need_weighted(nx_graph)
    
    def _generate_watts_strogatz(self) -> QAPGraphBase:
        """
        @article{
            watts1998collective,
            title={Collective dynamics of small-world networks},
            author={Watts, Duncan J and Strogatz, Steven H},
            journal={nature},
            volume={393},
            number={6684},
            pages={440--442},
            year={1998},
            publisher={Nature Publishing Group}
        }
        """
        # Generate Watts-Strogatz graph
        nx_graph: nx.Graph = nx.watts_strogatz_graph(
            n=self.nodes_num, k=self.ws_ring_neighbors, p=self.ws_prob
        )
        return self._if_need_weighted(nx_graph)
    
    def _generate_rb(self) -> QAPGraphBase:
        """
        @article{
            xu2005simple,
            title={A simple model to generate hard satisfiable instances},
            author={Xu, Ke and Boussemart, Fr{\'e}d{\'e}ric and Hemery, Fred and Lecoutre, Christophe},
            journal={arXiv preprint cs/0509032},
            year={2005}
        }
        """
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
        return self._if_need_weighted(nx_graph)

    def _if_need_weighted(self, nx_graph: nx.Graph) -> QAPGraphBase:
        """Assign weights to nodes and/or edges if required."""
        # Extract edge index
        edges = list(nx_graph.edges)
        edge_index = np.array(edges, dtype=np.int32).T

        # Add node feature if needed
        if self.node_weighted:
            node_feature = np.random.randn(self.nodes_num, self.node_feat_dim)
        else:
            node_feature = np.ones((self.nodes_num, 1), dtype=self.precision)

        # Add edge feature if needed
        edges_num = edge_index.shape[1]
        if self.edge_weighted:
            edge_feature = np.random.randn(edges_num, self.edge_feat_dim)
        else:
            edge_feature = np.ones((edges_num, 1), dtype=self.precision)

        # Create QAPGraphBase
        qap_graph = QAPGraphBase(precision=self.precision)
        qap_graph.from_data(
            node_feature=node_feature, 
            edge_feature=edge_feature, 
            edge_index=edge_index
        )
        return qap_graph

    def generate(self) -> QAPGraphBase:
        self.nodes_num = np.random.randint(self.nodes_num_min, self.nodes_num_max+1)
        generate_func = self.generate_func_dict[self.distribution_type]
        return generate_func()