r""""
Graph Matching(GM)

Graph matching aims to find a binary assignment matrix X that maximizes XᵀKX,
where K encodes feature-based similarities.
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
from typing import Union, Optional, Tuple
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.graphset.base import GraphSetTaskBase, Graph, get_pos_layer

class GMTask(GraphSetTaskBase):
    def __init__(
        self,
        graphs: list[Graph] = None,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Check graphs num
        if graphs is not None and len(graphs) != 2:
            raise ValueError("There must be two graphs.")
        
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.GM,
            minimize=False,
            graphs=graphs,
            precision=precision
        )
        
        
        self.aff_matrix: Optional[np.ndarray] = None
        
    def _deal_with_self_loop(self):
        if self.graphs is not None:
            for graph in self.graphs:
                graph.remove_self_loop()
                graph.self_loop = False
                
    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        n1, n2 = self.graphs[0].nodes_num, self.graphs[1].nodes_num
        X = sol.reshape(n1, n2)
        is_valid: bool = False
        if np.array_equal(X, X.astype(bool)):
            row_sum = X.sum(axis=1)   
            col_sum = X.sum(axis=0) 
            if (row_sum <= 1).all() or (col_sum <= 1).all(): 
                if n1 <= n2:            
                    is_valid = bool(np.all(row_sum==1))
                else:                    
                    is_valid = bool(np.all(col_sum == 1))
        return is_valid
    
    def from_data(
        self,
        graphs: list[Graph] = None,
        sol: np.ndarray = None,
        ref: bool = False,
        ):
        # Check num of graphs
        if graphs is not None and len(graphs) != 2:
            raise ValueError("There must be two graphs")
        
        super().from_data(graphs=graphs, sol=sol, ref=ref)
    
    def evaluate(self, sol:np.ndarray) -> float:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Evaluate
        if self.ref_sol is not None:
            mask = self.ref_sol == 1
            if not mask.any():
                return 0.0
            right = (sol[mask] == 1).sum()
            return float(right / mask.sum())
        else:
            if self.aff_matrix is not None:
                return float(sol.T @ self.aff_matrix @ sol.T)
            else:
                raise ValueError("Without ground-truth matching and affinity matrix")
    
    def render(
        self,
        save_path:  pathlib.Path,
        with_sol: bool = True,
        figsize: Tuple[float, float] = (10, 5),
        pos_type: str = "kamada_kawai_layout",
        node_color: str = "darkblue",
        matched_color: str = "orange",
        node_size: int = 30,
        edge_alpha: float = 0.5,
        edge_width: float = 1.0,
    ):
        check_file_path(save_path)
        sol = self.sol
        G1 = self.graphs[0].to_networkx()
        G2 = self.graphs[1].to_networkx()
        
        pos1 = get_pos_layer(pos_type)(G1)
        pos2 = get_pos_layer(pos_type)(G2)
        

        for k in pos1:
            pos1[k] = (pos1[k][0] - 1, pos1[k][1])
        for k in pos2:
            pos2[k] = (pos2[k][0] + 1, pos2[k][1])
        
        graph1_num = self.graphs[0].nodes_num
        graph2_num = self.graphs[1].nodes_num
            
        G2_shifted = nx.relabel_nodes(G2, lambda x: x + graph1_num)
        G = nx.compose(G1, G2_shifted)
        pos = {**pos1, **{k + graph1_num: v for (k, v) in pos2.items()}}    
            
        plt.figure(figsize=figsize)
        nx.draw(G, pos, node_color=node_color, node_size=node_size, alpha=edge_alpha)
        if with_sol:
            X = self.sol.reshape(graph1_num, graph2_num)
            matched = [(i, j + graph1_num) for i, j in zip(*np.where(X))]
            nx.draw_networkx_edges(
                G, pos, edgelist=matched, edge_color=matched_color,
                width=3, alpha=0.8, style="dashed"
            )
        plt.savefig(save_path, bbox_inches="tight")
    
    
    
            
