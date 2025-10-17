r"""
Minimum Vertex Cover (MVC).

MVC is to find the smallest subset of vertices in an undirected graph such that 
every edge in the graph is incident to at least one vertex in this subset.
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


import pathlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.graph.base import GraphTaskBase, get_pos_layer


class MVCTask(GraphTaskBase):
    def __init__(
        self, 
        node_weighted: bool = False, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.MVC, 
            minimize=True,
            node_weighted=node_weighted, 
            edge_weighted=False, 
            precision=precision
        )

    def _deal_with_self_loop(self):
        """Deal with self-loop."""
        self.remove_self_loop()
        self.self_loop = False

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        mask = self.edge_index[0] != self.edge_index[1]
        edge_index = self.edge_index[:, mask]
        src_dst = sol[edge_index[0]] + sol[edge_index[1]]
        return False if src_dst.min() == 0 else True
        
    def evaluate(self, sol: np.ndarray) -> np.floating:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Evaluate
        return np.sum(self.nodes_weight[sol.astype(np.bool_)])
        
    def render(
        self, 
        save_path: pathlib.Path, 
        with_sol: bool = True,
        figsize: tuple = (5, 5),
        pos_type: str = "kamada_kawai_layout",
        node_color: str = "darkblue",
        sol_node_color: str = "orange",
        node_size: int = 20,
        edge_color: str = "darkblue",
        edge_alpha: float = 0.5,
        edge_width: float = 1.0,
    ):
        """Render the MVC problem instance with or without solution."""
        
        # Check ``save_path``
        check_file_path(save_path)
        
        # Get Attributes
        sol = self.sol
        
        # Use ``to_networkx`` to get NetworkX graph
        nx_graph: nx.Graph = self.to_networkx()
                
        # Get Position Layer
        pos_layer = get_pos_layer(pos_type)
        pos = pos_layer(nx_graph)

        # Draw Graph
        figure = plt.figure(figsize=figsize)
        figure.add_subplot(111)
        if with_sol:
            if sol is None:
                raise ValueError("Solution is not provided!")
            colors = [sol_node_color if bit == 0 else node_color for bit in sol]
            nx.draw_networkx_nodes(
                G=nx_graph, pos=pos, node_color=colors, node_size=node_size
            )
        else:
            nx.draw_networkx_nodes(
                G=nx_graph, pos=pos, node_color=node_color, node_size=node_size
            )
        nx.draw_networkx_edges(
            G=nx_graph, pos=pos, edgelist=nx_graph.edges, 
            alpha=edge_alpha, width=edge_width, edge_color=edge_color
        )
        
        # Save Figure
        plt.savefig(save_path)