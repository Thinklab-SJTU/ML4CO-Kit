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
        node_aff_fn = None,
        edge_aff_fn = None,
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
        
        self.node_aff_fn = node_aff_fn
        self.edge_aff_fn = edge_aff_fn
        self.aff_mat: Optional[np.ndarray] = None
        
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
        
        if sol.shape != (n1, n2):
            return False
        
        if np.array_equal(sol, sol.astype(bool)):
            row_sum = sol.sum(axis=1)   
            col_sum = sol.sum(axis=0) 
            if (row_sum <= 1).all() or (col_sum <= 1).all(): 
                if n1 <= n2:            
                    is_valid = bool(np.all(row_sum==1))
                else:                    
                    is_valid = bool(np.all(col_sum == 1))
        return is_valid
    
    def inner_prod_aff_fn(self, feat1: np.ndarray, feat2: np.ndarray) -> np.ndarray:
        """inner product affinity function"""
        return np.matmul(feat1, feat2.T)
    
    def gaussian_aff_fn(self, feat1: np.ndarray, feat2: np.ndarray, sigma:np.floating = 1.0) -> np.ndarray:
         """Gaussian affinity function"""
         feat1 = np.expand_dims(feat1, axis=1)
         feat2 = np.expand_dims(feat2, axis=0)
         return np.exp(-((feat1-feat2)**2).sum(axis=-1)/sigma)
    
    def _aff_mat_from_node_edge_aff(self, node_aff: np.ndarray, edge_aff: np.ndarray, connectivity1: np.ndarray, connectivity2: np.ndarray,
                                n1, n2, ne1, ne2):
    
        if edge_aff is not None:
            dtype = edge_aff.dtype
            if n1 is None:
                n1 = np.amax(connectivity1).copy() + 1
            if n2 is None:
                n2 = np.amax(connectivity2).copy() + 1
            if ne1 is None:
                 ne1 = edge_aff.shape[0]
            if ne2 is None:
                ne2 = edge_aff.shape[1] 
        else:
            dtype = node_aff.dtype
            if n1 is None:
                n1 = node_aff.shape[0]
            if n2 is None:
                n2 = node_aff.shape[1]

    
        k = np.zeros((n2, n1, n2, n1), dtype=dtype)
        # edge-wise affinity
        if edge_aff is not None:
            edge_indices = np.concatenate([connectivity1.repeat(ne2, axis=0), np.tile(connectivity2, (ne1, 1))], axis=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]) # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[:ne1, :ne2].reshape(-1)
        k = k.reshape((n2 * n1, n2 * n1))
        # node-wise affinity
        if node_aff is not None:
            k[np.arange(n2 * n1), np.arange(n2 * n1)] = node_aff.T.reshape(-1)

        return k
    
    def build_aff_mat(
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
        node_aff_fn=None, 
        edge_aff_fn=None
        ):
        
        if node_feat1 is None:
            node_feat1 = self.graphs[0].node_feature
            n1 = self.graphs[0].nodes_num
        if node_feat2 is None:
            node_feat2 = self.graphs[1].node_feature
            n2 = self.graphs[1].nodes_num
        if edge_feat1 is None:
            edge_feat1 = self.graphs[0].edge_feature
            ne1 = self.graphs[0].edges_num
        if edge_feat2 is None:
            edge_feat2 = self.graphs[1].edge_feature
            ne2 = self.graphs[1].edges_num
        if connectivity1 is None:
            connectivity1 = self.graphs[0].edge_index.T
        if connectivity2 is None:
            connectivity2 = self.graphs[1].edge_index.T
    
      
        assert node_feat1 is not None and node_feat2 is not None, \
            'The following arguments must all be given if you want to compute node-wise affinity: ' \
            'node_feat1, node_feat2'
        assert edge_feat1 is not None and edge_feat2 is not None, \
            'The following arguments must all be given if you want to compute edge-wise affinity: ' \
            'edge_feat1, edge_feat2'
        
        if node_aff_fn is None:
            node_aff_fn = self.inner_prod_aff_fn
        if edge_aff_fn is None:
            edge_aff_fn = self.inner_prod_aff_fn
        
        node_aff = node_aff_fn(node_feat1, node_feat2) if node_feat1 is not None else None
        edge_aff = edge_aff_fn(edge_feat1, edge_feat2) if edge_feat1 is not None else None
        
        result = self._aff_mat_from_node_edge_aff(node_aff, edge_aff, connectivity1, connectivity2, n1, n2, ne1, ne2)
        
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
    
    def evaluate(self, sol:np.ndarray, mode: str = "acc") -> float:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Evaluate
        if mode == "acc":
            if self.ref_sol is not None:
                return (((self.ref_sol == 1) & (sol == 1)).sum() / (self.ref_sol == 1).sum()).astype(self.precision)
            else:
                raise ValueError("Without ground-truth matching solution")
        elif mode == "score":
            if self.aff_mat is not None:
                res = sol.T.ravel()
                return (res @ self.aff_mat @ res.T).astype(self.precision)
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
        matched_color: str = "orange",
        node_size: int = 30,
        edge_alpha: float = 0.5,
        edge_width: float = 1.0,
    ):
        check_file_path(save_path)
        G1 = self.graphs[0].to_networkx()
        G2 = self.graphs[1].to_networkx()
        
        pos1 = get_pos_layer(pos_type)(G1)
        pos2 = get_pos_layer(pos_type)(G2)
        

        for k in pos1:
            pos1[k] = (pos1[k][0] - 2, pos1[k][1])
        for k in pos2:
            pos2[k] = (pos2[k][0] + 2, pos2[k][1])
        
        graph1_num = self.graphs[0].nodes_num
            
        G2_shifted = nx.relabel_nodes(G2, lambda x: x + graph1_num)
        G = nx.compose(G1, G2_shifted)
        pos = {**pos1, **{k + graph1_num: v for (k, v) in pos2.items()}}    
            
        plt.figure(figsize=figsize)
        nx.draw(G, pos, node_color=node_color, node_size=node_size, alpha=edge_alpha, width=edge_width)
        if with_sol:
            X = self.sol if self.sol is not None else self.ref_sol
            matched = [(i, j + graph1_num) for i, j in zip(*np.where(X))]
            nx.draw_networkx_edges(
                G, pos, edgelist=matched, edge_color=matched_color,
                width=3, alpha=0.8, style="dashed"
            )
        plt.savefig(save_path, bbox_inches="tight")
    
    
    
            
