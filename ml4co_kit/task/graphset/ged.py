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
        node_match = None,
        node_subst_cost = None,
        node_del_cost = None,
        node_ins_cost = None,
        edge_match = None,
        edge_subst_cost = None,
        edge_del_cost = None,
        edge_ins_cost = None,
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
        self.node_math = node_match
        self.node_subst_cost = node_subst_cost
        self.node_ins_cost = node_ins_cost
        self.node_del_cost = node_del_cost
        self.edge_match = edge_match
        self.edge_subst_cost = edge_subst_cost
        self.edge_ins_cost = edge_ins_cost
        self.edge_del_cost = edge_del_cost
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
    
    def inner_prod_cost_fn(
        self, 
        feat1: np.ndarray, 
        feat2: np.ndarray, 
        ) -> np.ndarray:
        """inner product affinity function"""
        return -np.matmul(feat1, feat2.T)
    
    def gaussian_cost_fn(self, feat1: np.ndarray, feat2: np.ndarray, sigma:np.floating = 1.0) -> np.ndarray:
         """Gaussian affinity function"""
         feat1 = np.expand_dims(feat1, axis=1)
         feat2 = np.expand_dims(feat2, axis=0)
         return (1-np.exp(-((feat1-feat2)**2).sum(axis=-1)/sigma))
    
    def _cost_mat_from_node_edge_cost(self, node_cost: np.ndarray, edge_cost: np.ndarray, connectivity1: np.ndarray, connectivity2: np.ndarray,
                                n1, n2, ne1, ne2):
    
        if edge_cost is not None:
            dtype = edge_cost.dtype
            if n1 is None:
                n1 = np.amax(connectivity1).copy() + 1
            if n2 is None:
                n2 = np.amax(connectivity2).copy() + 1
            if ne1 is None:
                 ne1 = edge_cost.shape[0] - 1
            if ne2 is None:
                ne2 = edge_cost.shape[1] - 1 
        else:
            dtype = node_cost.dtype
            if n1 is None:
                n1 = node_cost.shape[0] - 1
            if n2 is None:
                n2 = node_cost.shape[1] - 1

    
        k = np.zeros((n2+1, n1+1, n2+1, n1+1), dtype=dtype)
        N = (n1 +1)* (n2 + 1)
        # edge-wise cost
        if edge_cost is not None:
            edge_indices = np.concatenate([connectivity1.repeat(ne2, axis=0), np.tile(connectivity2, (ne1, 1))], axis=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]) # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_cost[:ne1, :ne2].reshape(-1)
            # delect edge
            for i in range(ne1):
                k[n2, connectivity1[i][0], :, connectivity1[i][1]] = edge_cost[i, ne2]
                k[:, connectivity1[i][0], n2, connectivity1[i][1]] = edge_cost[i, ne2]
            # insert edge
            for i in range(ne2):
                k[connectivity2[i][0], :, connectivity2[i][1], n1] = edge_cost[ne1, i]
                k[connectivity2[i][0], n1, connectivity2[i][1], :] = edge_cost[ne1, i]
        k = k.reshape(N, N)
        # node-wise cost
        if node_cost is not None:
            k[np.arange(N), np.arange(N)] = node_cost.T.reshape(-1)
        return k
    
    def build_cost_mat(
        self,
        node_feat1: np.ndarray=None,
        edge_feat1: np.ndarray=None, 
        connectivity1: np.ndarray=None, 
        node_feat2: np.ndarray=None, 
        edge_feat2: np.ndarray=None, 
        connectivity2: np.ndarray=None,
        n1: int=None,
        ne1: int=None, 
        n2: int=None, 
        ne2: int=None,
        node_match = None,
        node_subst_cost = None,
        node_del_cost = None,
        node_ins_cost = None,
        edge_match = None,
        edge_subst_cost = None,
        edge_del_cost = None,
        edge_ins_cost = None
        ):
        
        if node_feat1 is None:
            node_feat1 = self.graphs[0].nodes_feature
            n1 = self.graphs[0].nodes_num
        if node_feat2 is None:
            node_feat2 = self.graphs[1].nodes_feature
            n2 = self.graphs[1].nodes_num
        if edge_feat1 is None:
            edge_feat1 = self.graphs[0].edges_feature
            ne1 = self.graphs[0].edges_num
        if edge_feat2 is None:
            edge_feat2 = self.graphs[1].edges_feature
            ne2 = self.graphs[1].edges_num
        if connectivity1 is None:
            connectivity1 = self.graphs[0].edge_index.T
        if connectivity2 is None:
            connectivity2 = self.graphs[1].edge_index.T
    
      
        assert node_feat1 is not None and node_feat2 is not None, \
            'The following arguments must all be given if you want to compute node-wise cost: ' \
            'node_feat1, node_feat2'
        assert edge_feat1 is not None and edge_feat2 is not None, \
            'The following arguments must all be given if you want to compute edge-wise affinity: ' \
            'edge_feat1, edge_feat2'
        
        if node_subst_cost is None:
            node_subst_cost = self.gaussian_cost_fn
        if edge_subst_cost is None:
            edge_subst_cost = self.gaussian_cost_fn
            
        # Node cost 
        node_subst_cost_mat = node_subst_cost(node_feat1, node_feat2)

        if node_match is not None:
            node_subst_cost_mat = node_subst_cost_mat * node_match(node_feat1, node_feat2)
        if node_ins_cost is not None:
            node_ins_cost_vec = node_ins_cost(node_feat2)  
        else:
            node_ins_cost_vec = 1-np.exp(-(node_feat2**2).sum(axis=-1)/1.0)

        if node_del_cost is not None:
            node_del_cost_vec = node_del_cost(node_feat1) 
        else:
            node_del_cost_vec = 1-np.exp(-(node_feat1**2).sum(axis=-1)/1.0)

        node_cost = np.block(
            [
            [node_subst_cost_mat, node_del_cost_vec[:, None]],
            [node_ins_cost_vec[None, :], np.array([[0]])]
            ]
        ).astype(self.precision)
        
        # Edge cost
        edge_subst_cost_mat = edge_subst_cost(edge_feat1, edge_feat2)
        if edge_match is not None:
            edge_subst_cost_mat = edge_subst_cost_mat * edge_match(edge_feat1, edge_feat2)
      
        if edge_ins_cost is not None:
            edge_ins_cost_vec = edge_ins_cost(edge_feat2)
        else:
            edge_ins_cost_vec =  1-np.exp(-(edge_feat2**2).sum(axis=-1)/1.0)
            
        if edge_del_cost is not None:
            edge_del_cost_vec = edge_del_cost(edge_feat1)
        else:
            edge_del_cost_vec = 1-np.exp(-(edge_feat1**2).sum(axis=-1)/1.0)
       
        edge_cost = np.block(
            [
            [edge_subst_cost_mat, edge_del_cost_vec[:, None]],
            [edge_ins_cost_vec[None, :], np.array([0])]
            ]
        ).astype(self.precision)
        
        # print("node substitution cost matrix:")
        # print(node_subst_cost_mat)
        # print("node insertion cost vector:")
        # print(node_ins_cost_vec)
        # print("node deletion cost vector:")
        # print(node_del_cost_vec)
        # print("edge substitution cost matrix:")
        # print(edge_subst_cost_mat)
        # print("edge insertion cost vector:")
        # print(edge_ins_cost_vec)
        # print("edge deletion cost vector:")
        # print(edge_del_cost_vec)
        
        
        
        result = self._cost_mat_from_node_edge_cost(node_cost, edge_cost, connectivity1, connectivity2, n1, n2, ne1, ne2)
        
        return result 
    
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
            return (((self.ref_sol == 1) & (sol == 1)).sum() / (self.ref_sol == 1).sum()).astype(self.precision)
        else:
            if self.cost_mat is not None:
                res = sol.T.ravel()
                return (res @ self.cost_mat @ res.T).astype(self.precision)
            else:
                raise ValueError("Without ground-truth matching and affinity matrix")
    
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
    
    
    
            
