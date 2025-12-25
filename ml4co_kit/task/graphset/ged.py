r""""
Graph Edit Distance

Graph Edit Distance aim to find both the minimum total cost and the corresponding edit path 
required to transform graph G₁ into graph G₂ through a sequence of basic edit operations,
including node/edge insertion, node/edge deletion, and node/edge substitution.
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


class GEDTask(GraphSetTaskBase):
    def __init__(
        self,
        graphs: list[Graph] = None,
        node_mapping: np.array = None, # substitution_with_same_label, substitution_with_diff_label, insert, delete cost
        edge_mapping: np.array = None, # substitution, insert, delete cost
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Check graphs num
        if graphs is not None and len(graphs) != 2:
            raise ValueError("There must be two graphs.")
        
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.GED,
            minimize=False,
            graphs=graphs,
            precision=precision
        )
        
        self.graphs = graphs
        self.node_mapping = node_mapping
        self.edge_mapping = edge_mapping
        self.cost_mat: Optional[np.ndarray] = None # ((n1+1)*(n2+1), (n1+1)*(n2+1))
        
    def _deal_with_self_loop(self):
        if self.graphs is not None:
            for graph in self.graphs:
                graph.remove_self_loop()
                graph.self_loop = False

    def _check_sol_dim(self):
        """Ensure solution is a 2D array."""
        if self.sol.ndim != 2:
            raise ValueError("Solution should be a 2D array.")
        
    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 2D array."""
        if self.ref_sol.ndim != 2:
            raise ValueError("Reference solution should be a 2D array.")
        
    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        is_valid: bool = False
        n1 = self.graphs[0].nodes_num
        n2 = self.graphs[1].nodes_num
        
        if sol.shape != (n1 + 1, n2 + 1):
            return False
        
        if np.array_equal(sol, sol.astype(bool)):
            row_sum = sol[:n1, :].sum(axis=1)   
            col_sum = sol[:, :n2].sum(axis=0) 
            is_valid = bool((row_sum == 1).all() and (col_sum == 1).all())
        return is_valid
 
    def _node_cost_fn(self, dummy_node1: np.ndarray, dummy_node2: np.ndarray, mapping: np.array = None, threshold: float = 0.1) -> np.ndarray:
        if mapping is not None:
            if len(mapping) != 4:
                raise ValueError("Node mapping must have four elements.")
        else:
            mapping = np.array([0, 1, 1, 1])
            
        thresh = self.precision(threshold)
        dim = dummy_node1.shape[1]
        dist = np.sum(np.abs(dummy_node1[:, None, :] - dummy_node2[None, :, :]), axis=-1).astype(self.precision)
        dist /= dim
        dist = (dist > thresh).astype(np.int32)
        dist[-1, :] = 2  
        dist[:, -1] = 3 
        dist[-1, -1] = 0
        return mapping[dist]    
         
    def _edge_cost_fn(self, dummy_adj1: np.ndarray, dummy_adj2: np.ndarray, mapping: np.array = None) -> np.ndarray:
        if mapping is not None:
            if len(mapping) != 3:
                raise ValueError("Edge mapping must have three elements.")
        else:
            mapping = np.array([0, 1, 1])
            
        a1 = dummy_adj1.reshape(-1, 1)
        a2 = dummy_adj2.reshape(1, -1)
        dist = (a1 - a2 + 1).astype(np.int32)
        
        if mapping is None:
            mapping = np.array([0, 1, 1])
            
        _mapping = mapping[[1, 0, 2]]  # insert, substitution, delete
        k = _mapping[dist]
        k = k.reshape(dummy_adj1.shape[0], dummy_adj1.shape[1], dummy_adj2.shape[0], dummy_adj2.shape[1])
        k = k.transpose(0, 2, 1, 3).reshape(dummy_adj1.shape[0]*dummy_adj2.shape[0], dummy_adj1.shape[1]*dummy_adj2.shape[1])
        
        return k/2
        
    def build_cost_mat(self, node_mapping: np.array = None, edge_mapping: np.array = None) -> np.ndarray:
        """Build cost matrix from node and edge features."""
        if self.graphs is None:
            raise ValueError("Graphs are not provided.")
        
        if node_mapping is None:
            node_mapping = self.node_mapping
        if edge_mapping is None:
            edge_mapping = self.edge_mapping
            
        g1 = self.graphs[0]
        g2 = self.graphs[1]
        
        
        adj1 = np.zeros((g1.nodes_num, g1.nodes_num), dtype=self.precision)
        for edge in g1.edge_index.T:
            adj1[edge[0], edge[1]] = 1
            adj1[edge[1], edge[0]] = 1
            
        adj2 = np.zeros((g2.nodes_num, g2.nodes_num), dtype=self.precision)
        for edge in g2.edge_index.T:
            adj2[edge[0], edge[1]] = 1
            adj2[edge[1], edge[0]] = 1
        
        dummy_node1 = np.vstack([g1.node_feature, np.zeros((1, g1.node_feature.shape[1]), dtype=self.precision)])
        dummy_node2 = np.vstack([g2.node_feature, np.zeros((1, g2.node_feature.shape[1]), dtype=self.precision)])   
        dummy_adj1 = np.vstack([np.hstack([adj1, np.zeros((g1.nodes_num, 1), dtype=np.int32)]),
                                np.zeros((1, g1.nodes_num + 1), dtype=np.int32)])
        dummy_adj2 = np.vstack([np.hstack([adj2, np.zeros((g2.nodes_num, 1), dtype=np.int32)]),
                                np.zeros((1, g2.nodes_num + 1), dtype=np.int32)])
        
        node_cost = self._node_cost_fn(dummy_node1, dummy_node2, mapping=node_mapping)
        k = self._edge_cost_fn(dummy_adj1, dummy_adj2, mapping=edge_mapping)
        
        k[np.arange(k.shape[0]), np.arange(k.shape[0])] = node_cost.reshape(-1) 
        
        self.cost_mat = k.astype(self.precision)
        
        return self.cost_mat
           
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
    
    def evaluate(self, sol:np.ndarray, mode: str = "cost") -> float:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Evaluate
        if mode == "acc":
           if self.ref_sol is not None:
               return (((self.ref_sol == 1) & (sol == 1)).sum() / (self.ref_sol == 1).sum()).astype(self.precision)
           else:
               raise ValueError("Without ground-truth edit path")
        elif mode == "cost":
            if self.cost_mat is not None:
                res = sol.ravel()
                return (res @ self.cost_mat @ res.T).astype(self.precision)
            else:
                raise ValueError("Without cost matrix")
        else:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
    
    def render(
        self,
        save_path:  pathlib.Path,
        with_sol: bool = True,
        figsize: Tuple[float, float] = (10, 5),
        pos_type: str = "kamada_kawai_layout",
        node_color: str = "darkblue",
        dummy_node_color: str = "green",
        matched_color: str = "orange",
        dummy_matched_color: str = "red",
        node_size: int = 30,
        edge_alpha: float = 0.5,
        edge_width: float = 1.0,
    ):
        check_file_path(save_path)
        G1 = self.graphs[0].to_networkx()
        G2 = self.graphs[1].to_networkx()
        graph1_num = self.graphs[0].nodes_num
        graph2_num = self.graphs[1].nodes_num
        
        G1.add_node(graph1_num)  # add dummy node 
        G2.add_node(graph2_num)  # add dummy node
        
        pos1 = get_pos_layer(pos_type)(G1)
        pos2 = get_pos_layer(pos_type)(G2)
        

        for k in pos1:
            pos1[k] = (pos1[k][0] - 2, pos1[k][1])
        for k in pos2:
            pos2[k] = (pos2[k][0] + 2, pos2[k][1])
        
        G2_shifted = nx.relabel_nodes(G2, lambda x: x + graph1_num + 1)
        G = nx.compose(G1, G2_shifted)
        pos = {**pos1, **{k + graph1_num + 1: v for (k, v) in pos2.items()}}   
        dummy_nodes = [graph1_num, graph2_num + graph1_num + 1]   
            
        plt.figure(figsize=figsize)
        nx.draw(G, pos, node_color=node_color, node_size=node_size, alpha=edge_alpha, width=edge_width)   
        nx.draw_networkx_nodes(G, pos, nodelist=dummy_nodes, node_color=dummy_node_color, node_size=node_size)
        if with_sol:
            X = (self.sol if self.sol is not None else self.ref_sol).astype(bool)
            idx = np.argwhere(X)     
            matched = []
            edge_colors = []
            for i, j in idx:
                matched.append((i, j + graph1_num + 1))
                edge_colors.append(dummy_matched_color if i == graph1_num or j == graph2_num else matched_color)

            nx.draw_networkx_edges(
                G, pos,
                edgelist=matched,
                edge_color=edge_colors,
                width=3,
                alpha=0.8,
                style="dashed"
            )
        
        plt.savefig(save_path, bbox_inches="tight")
    
    
    
            
