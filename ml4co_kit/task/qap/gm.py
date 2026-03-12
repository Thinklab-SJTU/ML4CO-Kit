r"""
Graph Matching Task.

Graph Matching aim to find the optimal matching between two graphs, 
where the optimal matching is the one that maximizes the similarity between the two graphs.
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


import numpy as np
from typing import Union, Callable, Dict, Sequence
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.base import QAPTaskBase, QAPGraphBase


class GMAffinityMatrixBuilder(object):
    def __init__(
        self,
        node_aff_fn: Union[Callable, str] = "inner_product",
        edge_aff_fn: Union[Callable, str] = "inner_product",
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(GMAffinityMatrixBuilder, self).__init__()
        
        # Affinity Function Dictionary
        aff_fn_dict: Dict[str, Callable] = {
            "inner_product": self._inner_prod_aff_fn,
            "gaussian": self._gaussian_aff_fn
        }

        # Node Affinity Function
        if isinstance(node_aff_fn, str):
            self.node_aff_fn = aff_fn_dict[node_aff_fn]
        else:
            self.node_aff_fn = node_aff_fn

        # Edge Affinity Function
        if isinstance(edge_aff_fn, str):
            self.edge_aff_fn = aff_fn_dict[edge_aff_fn]
        else:
            self.edge_aff_fn = edge_aff_fn

        # Precision
        self.precision = precision

    def _inner_prod_aff_fn(
        self, 
        feat1: np.ndarray = None, 
        feat2: np.ndarray = None, 
    ) -> np.ndarray:
        """inner product affinity function"""
        if feat1 is None:
            return None
        else:
            return np.matmul(feat1, feat2.T)
    
    def _gaussian_aff_fn(
        self, 
        feat1: np.ndarray = None, 
        feat2: np.ndarray = None, 
        sigma: np.floating = 1.0
    ) -> np.ndarray:
        """Gaussian affinity function"""
        if feat1 is None:
            return None
        else:
            feat1 = np.expand_dims(feat1, axis=1)
            feat2 = np.expand_dims(feat2, axis=0)
            return np.exp(-((feat1 - feat2)**2).sum(axis=-1) / sigma)

    def build_aff_mat(self, g1: QAPGraphBase, g2: QAPGraphBase) -> np.ndarray:
        # Build Node Affinity
        node_aff: np.ndarray = self.node_aff_fn(g1.node_feature, g2.node_feature)
        
        # Build Edge Affinity
        edge_aff: np.ndarray = self.edge_aff_fn(g1.edge_feature, g2.edge_feature)

        # Build Affinity Matrix (Initialize)
        n1 = g1.nodes_num
        n2 = g2.nodes_num
        ne1 = g1.edges_num
        ne2 = g2.edges_num
        n1n2 = n1 * n2
        conn1 = g1.edge_index.T
        conn2 = g2.edge_index.T
        K = np.zeros(shape=(n2, n1, n2, n1), dtype=self.precision)

        # Build Affinity Matrix (Edge-wise)
        if edge_aff is not None:
            edge_indices = np.concatenate(
                [conn1.repeat(ne2, axis=0), np.tile(conn2, (ne1, 1))], axis=1
            ) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (
                edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]
            ) # indices: start_g2, start_g1, end_g2, end_g1
            K[edge_indices] = edge_aff[:ne1, :ne2].reshape(-1)
        
        # Build Affinity Matrix (Node-wise)
        K = K.reshape((n1n2, n1n2))
        if node_aff is not None:
            K[np.arange(n1n2), np.arange(n1n2)] = node_aff.T.reshape(-1)

        # import pdb
        # pdb.set_trace()
        # import pygmtools as pygm
        # pygm.set_backend('numpy')
        # pygm_K = pygm.utils.build_aff_mat(
        #     node_feat1=np.expand_dims(g1.node_feature, axis=0), 
        #     edge_feat1=np.expand_dims(g1.edge_feature, axis=0), 
        #     connectivity1=np.expand_dims(g1.edge_index.T, axis=0), 
        #     node_feat2=np.expand_dims(g2.node_feature, axis=0), 
        #     edge_feat2=np.expand_dims(g2.edge_feature, axis=0), 
        #     connectivity2=np.expand_dims(g2.edge_index.T, axis=0), 
        #     n1=np.array([n1]), 
        #     n2=np.array([n2]), 
        #     edge_aff_fn=pygm.utils.inner_prod_aff_fn
        # )
        # pdb.set_trace()
        # X = pygm.rrwm(K, n1, n2, beta=100)
        return K
    
    def __repr__(self):
        return f"GMAffinityMatrixBuilder(node_aff_fn={self.node_aff_fn.__name__}, edge_aff_fn={self.edge_aff_fn.__name__}, precision={self.precision})"


class GMTask(QAPTaskBase):
    def __init__(
        self,
        affn_builder: GMAffinityMatrixBuilder = GMAffinityMatrixBuilder(),
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(GMTask, self).__init__(
            task_type=TASK_TYPE.GM, 
            minimize=False, 
            precision=precision
        )

        # Initialize Attributes
        self.affn_builder = affn_builder # affinity matrix builder
        self.g1 = None    # graph 1
        self.g2 = None    # graph 2

    def from_data(
        self, 
        g1: QAPGraphBase = None, 
        g2: QAPGraphBase = None,
        sol: np.ndarray = None,
        ref: bool = False,
    ) -> None:
        # Set Attributes
        if g1 is not None:
            self.g1 = g1
            n1 = g1.nodes_num
        else:
            n1 = None
        if g2 is not None:
            self.g2 = g2
            n2 = g2.nodes_num
        else:
            n2 = None
            
        # Build Affinity Matrix
        if g1 is not None or g2 is not None:
            K = self.affn_builder.build_aff_mat(g1, g2)
        else:
            K = None

        # Call super ``from_data``
        super().from_data(K=K, n1=n1, n2=n2, sol=sol, ref=ref)

    def evaluate_w_gap(self, mode: str = "acc") -> Sequence[np.floating]:
        """Evaluate the given solution with gap."""
        # Check if the solution and reference solution are not None
        if self.sol is None or self.ref_sol is None:
            raise ValueError("Solution and reference solution cannot be None!")
        
        # Evaluate the solution and reference solution
        sol_cost = self.evaluate(self.sol, mode=mode)
        ref_cost = self.evaluate(self.ref_sol, mode=mode)

        # Calculate the gap
        if abs(ref_cost) < 1e-8:
            gap = None
        else:
            if self.minimize:
                gap = (sol_cost - ref_cost) / ref_cost
            else:
                gap = (ref_cost - sol_cost) / ref_cost
            gap = gap * np.array(100.0).astype(self.precision)
        
        return sol_cost, ref_cost, gap