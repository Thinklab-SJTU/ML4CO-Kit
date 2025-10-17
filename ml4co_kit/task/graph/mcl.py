r"""
Maximum Clique (MCl).

MCl is to find the largest complete subgraph in an undirected graph, 
where every pair of vertices in the subgraph is connected by an edge.
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


class MClTask(GraphTaskBase):
    def __init__(
        self, 
        node_weighted: bool = False, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.MCL, 
            minimize=False,
            node_weighted=node_weighted, 
            edge_weighted=False, 
            precision=precision
        )

    def _deal_with_self_loop(self):
        """Deal with self-loop."""
        self.add_self_loop()
        self.self_loop = True
    
    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        index = np.where(sol == 1)[0]
        adj_matrix = self.to_adj_matrix()
        np.fill_diagonal(adj_matrix, 1)
        return True if adj_matrix[index][:, index].all() else False
        
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
        """Render the MCl problem instance with or without solution."""
        
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
            colors = [node_color if bit == 0 else sol_node_color for bit in sol]
            nx.draw_networkx_nodes(G=nx_graph, pos=pos, node_color=colors, node_size=node_size)
            
            # Separate edges into clique and non-clique edges
            clique_nodes = set(np.where(sol == 1)[0])  # Nodes in the maximum clique
            clique_edges = [(u, v) for u, v in nx_graph.edges if u in clique_nodes and v in clique_nodes]
            other_edges = [(u, v) for u, v in nx_graph.edges if (u, v) not in clique_edges]
            
            # Draw Clique Edges
            nx.draw_networkx_edges(
                G=nx_graph, pos=pos, edgelist=clique_edges, alpha=edge_alpha, 
                width=edge_width, edge_color=edge_color
            )
            
            # Draw Non-Clique Edges
            nx.draw_networkx_edges(
                G=nx_graph, pos=pos, edgelist=other_edges, alpha=0.5, 
                width=0.5, edge_color="gray"
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