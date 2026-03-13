r"""
Base Task Class for Graph Set Problems.
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
import scipy.sparse
from typing import Union
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class QAPGraphBase(object):
    def __init__(
        self,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Initialize Attributes (Node)   
        self.nodes_num: int = None            
        self.node_feature: np.ndarray = None
        self.node_feat_dim: int = None

        # Initialize Attributes (Edge)   
        self.edges_num: int = None
        self.edge_feature: np.ndarray = None
        self.edge_feat_dim: int = None
        self.edge_index: np.ndarray = None
        self.adj_matrix: np.ndarray = None

        # Initialize Attributes (Precision)
        self.precision = precision

    def from_data(
        self, 
        node_feature: np.ndarray = None, 
        edge_feature: np.ndarray = None, 
        edge_index: np.ndarray = None
    ):
        # Node Feature (V, F)
        if node_feature is not None:
            # Check if the node feature is a 2D array
            if node_feature.ndim == 1:
                node_feature = node_feature.reshape(-1, 1)
            if node_feature.ndim != 2:
                raise ValueError(
                    "Node feature should be a 2D array with shape (nodes_num, node_feat_dim)."
                )

            # Set Attributes
            self.node_feature = node_feature.astype(self.precision)
            self.node_feat_dim = self.node_feature.shape[1]
            self.nodes_num = int(node_feature.shape[0])
        
        # Edge Feature (E, F)
        if edge_feature is not None:
            # Check if the edge feature is a 2D array
            if edge_feature.ndim == 1:
                edge_feature = edge_feature.reshape(-1, 1)
            if edge_feature.ndim != 2:
                raise ValueError(
                    "Edge feature should be a 2D array with shape (edges_num, edge_feat_dim)."
                )

            # Set Attributes
            self.edge_feature = edge_feature.astype(self.precision)
            self.edge_feat_dim = self.edge_feature.shape[1]
            self.edges_num = int(edge_feature.shape[0])
        
        # Edge Index (2, E)
        if edge_index is not None:
            self.edge_index = edge_index
            # Update edges_num if not set by edge_feature
            if self.edges_num is None:
                self.edges_num = int(edge_index.shape[1])

    def to_adj_matrix(self) -> np.ndarray:
        """Convert edge_index to adjacency matrix."""
        if self.adj_matrix is None:
            self.adj_matrix = scipy.sparse.coo_matrix(
                arg1=(
                    np.ones(self.edges_num), 
                    (self.edge_index[0], self.edge_index[1])
                ), 
                shape=(self.nodes_num, self.nodes_num)
            ).toarray()
        return self.adj_matrix
    
    def make_symmetric(self):
        # Step 1: Check if edge_index is already symmetric
        adj_matrix = self.to_adj_matrix()
        
        # Check if adjacency matrix is symmetric
        if np.array_equal(adj_matrix, adj_matrix.T):
            # Already symmetric, just mark it
            return
        
        # Step 2: Create reverse edges by flipping edge_index
        # edge_index shape: (2, num_edges), flip rows to get reverse edges
        reverse_edge_index = np.flip(self.edge_index, axis=0)
        
        # Step 3: Concatenate original and reverse edges
        symmetric_edge_index = np.hstack((self.edge_index, reverse_edge_index))
        
        # Step 4: Handle edge features (if they exist)
        if self.edge_feature is not None:
            # Duplicate edge features for reverse edges
            symmetric_edge_feature = np.vstack((self.edge_feature, self.edge_feature))
        else:
            symmetric_edge_feature = None
        
        # Step 5: Remove duplicate edges (in case some reverse edges already existed)
        # Create a set of unique edges
        edges_set = set()
        unique_indices = []
        
        for i in range(symmetric_edge_index.shape[1]):
            edge = (symmetric_edge_index[0, i], symmetric_edge_index[1, i])
            if edge not in edges_set:
                edges_set.add(edge)
                unique_indices.append(i)
        
        # Filter to keep only unique edges
        symmetric_edge_index = symmetric_edge_index[:, unique_indices]
        if symmetric_edge_feature is not None:
            symmetric_edge_feature = symmetric_edge_feature[unique_indices]
        
        # Step 6: Update graph with symmetric structure
        self.edges_num = None
        self.edge_feature = None
        self.adj_matrix = None
        
        # Update data
        self.from_data(
            node_feature=self.node_feature,
            edge_feature=symmetric_edge_feature,
            edge_index=symmetric_edge_index
        )
    
    def __repr__(self):
        return (
            "QAPGraphBase("
            f"nodes_num={self.nodes_num}, "
            f"edges_num={self.edges_num}, "
            f"node_feat_dim={self.node_feat_dim}, "
            f"edge_feat_dim={self.edge_feat_dim}, "
            f"precision={self.precision.__name__}"
            ")"
        )
    

class QAPTaskBase(TaskBase):
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(QAPTaskBase, self).__init__(
            task_type=task_type, 
            minimize=minimize, 
            precision=precision
        )

        # Initialize Attributes
        self.K = None     # affinity matrix (n1*n2, n1*n2)
        self.n1 = None    # number of nodes in graph 1
        self.n2 = None    # number of nodes in graph 2

    def _check_sol_dim(self):
        """Ensure solution is a 2D array."""
        if self.sol.ndim != 2:
            raise ValueError("Solution should be a 2D array.")

    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 2D array."""
        if self.ref_sol.ndim != 2:
            raise ValueError("Reference solution should be a 2D array.")

    def _check_K_n1_n2(self):
        k_shape = self.K.shape
        n1n2 = self.n1 * self.n2
        if k_shape[0] != n1n2 or k_shape[1] != n1n2:
            raise ValueError(
                f"Affinity matrix should have shape (n1*n2, n1*n2), "
                f"that is ({n1n2}, {n1n2}), but got ({k_shape[0]}, {k_shape[1]})."
            )

    def from_data(
        self, 
        K: np.ndarray = None, 
        n1: int = None, 
        n2: int = None,
        sol: np.ndarray = None,
        ref: bool = False,
    ) -> None:
        # Set Attributes
        if K is not None:
            self.K = K
        if n1 is not None:
            self.n1 = n1
        if n2 is not None:
            self.n2 = n2

        # Set Solution if Provided
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()

        # Check if the affinity matrix is valid
        self._check_K_n1_n2()

    def check_constraints(self, sol: np.ndarray) -> bool:
        # Check if the solution is a 2D array with shape (n1, n2)
        sol_shape = sol.shape
        if sol_shape != (self.n1, self.n2):
            return False

        # Check if the solution is a valid assignment matrix
        if not np.array_equal(sol, sol.astype(np.int32)):
            return False

        # Check col and row sum
        row_sum: np.ndarray = sol.sum(axis=1)
        col_sum: np.ndarray = sol.sum(axis=0)
        if not (row_sum <= 1).all() or not (col_sum <= 1).all():
            return False

        return True

    def evaluate(self, sol: np.ndarray, mode: str = "acc") -> np.floating:
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")

        # Evaluate
        if mode == "acc":
            self._check_ref_sol_not_none()
            acc = (sol * self.ref_sol).sum() / self.ref_sol.sum()
            return acc
        
        elif mode == "score":
            sol_ravel = sol.T.ravel()
            score = sol_ravel.T @ self.K @ sol_ravel
            return score
        
        else:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
