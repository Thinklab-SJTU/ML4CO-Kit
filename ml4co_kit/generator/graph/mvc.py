r"""
Generator for Minimum Vertex Cover (MVC) instances.
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
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.graph.mvc import MVCTask
from ml4co_kit.generator.graph.base import GRAPH_TYPE
from ml4co_kit.generator.graph.base import (
    GraphGeneratorBase, GRAPH_WEIGHT_TYPE, GraphWeightGenerator
)


class MVCGenerator(GraphGeneratorBase):
    """Generator for Minimum Vertex Cover (MVC) instances."""
    
    def __init__(
        self,
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
        # special args for weighted graph (node weights)
        node_weighted: bool = False,
        node_weighted_gen: GraphWeightGenerator = GraphWeightGenerator(
            weighted_type=GRAPH_WEIGHT_TYPE.UNIFORM),
    ):
        # Super Initialization
        super(MVCGenerator, self).__init__(
            task_type=TASK_TYPE.MVC, 
            distribution_type=distribution_type, 
            precision=precision,
            nodes_num_scale=nodes_num_scale,
            er_prob=er_prob,
            ba_conn_degree=ba_conn_degree,
            hk_prob=hk_prob,
            hk_conn_degree=hk_conn_degree,
            ws_prob=ws_prob,
            ws_ring_neighbors=ws_ring_neighbors,
            rb_n_scale=rb_n_scale,
            rb_k_scale=rb_k_scale,
            rb_p_scale=rb_p_scale,
            node_weighted=node_weighted,
            node_weighted_gen=node_weighted_gen,
            edge_weighted=False # MVC does not support edge weights   
        )
        
    def _create_instance(self, nx_graph: nx.Graph) -> MVCTask:
        data = MVCTask(
            node_weighted=self.node_weighted,
            precision=self.precision
        )
        data.from_networkx(nx_graph)
        data.remove_self_loop()
        return data