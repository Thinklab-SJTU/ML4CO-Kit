r"""
Maximum Cut (MCut). 

MCut involves partitioning the vertices of an undirected graph into 
two disjoint sets to maximize the number of edges crossing between them.
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


class MCutTask(GraphTaskBase):
    def __init__(
        self, 
        edge_weighted: bool = False, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.MCUT, 
            minimize=False,
            node_weighted=False,
            edge_weighted=edge_weighted,  
            precision=precision
        )

    def _deal_with_self_loop(self):
        """Deal with self-loop."""
        self.remove_self_loop()
        self.self_loop = False

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        return True
        
    def evaluate(self, sol: np.ndarray) -> np.floating:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")

        # Evaluate
        src_index = sol[self.edge_index[0]] 
        dst_index = sol[self.edge_index[1]]
        mask = src_index != dst_index
        return np.sum(self.edges_weight[mask]) / 2

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
        """Render the MCut problem instance with or without solution."""
        
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
            nx.draw_networkx_nodes(
                G=nx_graph, pos=pos, node_color=colors, node_size=node_size
            )
            
            # Different edge states
            sel_nodes = set(np.where(sol == 1)[0])
            cut_edges = list()
            other_edges = list()
            for u, v in nx_graph.edges:
                if (u in sel_nodes and v not in sel_nodes) or (v in sel_nodes and u not in sel_nodes):
                    cut_edges.append((u, v))
                else:
                    other_edges.append((u, v))
            
            # Draw Graph
            nx.draw_networkx_edges(
                G=nx_graph, pos=pos, edgelist=cut_edges, alpha=edge_alpha, 
                width=edge_width, edge_color=edge_color
            )
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