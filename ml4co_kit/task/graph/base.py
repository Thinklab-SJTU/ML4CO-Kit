r"""
Base Task Class for Graph Problems.
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


import pickle
import pathlib
import numpy as np
import scipy.sparse
import networkx as nx
from typing import Union, Tuple
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path


class GraphTaskBase(TaskBase):
    r"""
    Base class for all undirected graph problems in the ML4CO kit.
    """
    
    def __init__(
        self, 
        task_type: TASK_TYPE,
        minimize: bool,
        node_weighted: bool = False,
        edge_weighted: bool = False,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(GraphTaskBase, self).__init__(
            task_type=task_type, 
            minimize=minimize,
            precision=precision
        )
        
        # Initialize Attributes (basic)
        self.node_weighted = node_weighted
        self.edge_weighted = edge_weighted
        self.nodes_num: int = None
        self.edges_num: int = None
        self.nodes_weight: np.ndarray = None
        self.edges_weight: np.ndarray = None
        self.edge_index: np.ndarray = None # [Method 1] Edge Index in shape (2, num_edges)
        
        # Symmetric
        self.already_symmetric = False
        
        # Self-loop
        self.self_loop = None
        
        # Initialize Attributes (other structure)
        self.adj_matrix: np.ndarray = None           # [Method 2] Adjacency Matrix
        self.adj_matrix_weighted: np.ndarray = None  # [Method 3] Adjacency Matrix with edges_weight
        self.xadj: np.ndarray = None                 # [Method 4] Compressed Sparse Row (CSR) representation
        self.adjncy: np.ndarray = None               # [Method 4] Compressed Sparse Row (CSR) representation
    
    def _check_nodes_weight_dim(self):
        """Ensure node weights is a 1D array."""
        if self.nodes_weight.ndim != 1:
            raise ValueError("Node weights should be a 1D array with shape (num_nodes,).")
    
    def _check_edges_weight_dim(self):
        """Ensure edge weights is a 1D array."""
        if self.edges_weight.ndim != 1:
            raise ValueError("Edge weights should be a 1D array with shape (num_edges,).")
    
    def _check_edges_index_dim(self):
        """Ensure edge index is a 2D array with shape (2, num_edges)."""
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("Edge index should be a 2D array with shape (2, num_edges).")
        if self.edges_num is not None and self.edge_index.shape[1] != self.edges_num:
            raise ValueError("Edge index second dimension should match number of edges.")

    def _check_edges_index_not_none(self):
        """Ensure edge index is not None."""
        if self.edge_index is None:
            raise ValueError("Edge index cannot be None!")

    def _check_sol_dim(self):
        """Ensure solution is a 1D array."""
        if self.sol.ndim != 1:
            raise ValueError("Solution should be a 1D array.")
        
    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 1D array."""
        if self.ref_sol.ndim != 1:
            raise ValueError("Reference solution should be a 1D array.")
    
    def _invalidate_cached_structures(self):
        """Invalidate cached structures."""
        self.adj_matrix = None
        self.adj_matrix_weighted = None
        self.xadj = None
        self.adjncy = None
    
    def _deal_with_self_loop(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def from_data(
        self,
        edge_index: np.ndarray = None,
        nodes_weight: np.ndarray = None, 
        edges_weight: np.ndarray = None,
        sol: np.ndarray = None,
        ref: bool = False,
    ):
        # Set Attributes and Check Dimensions
        if nodes_weight is not None:
            if self.node_weighted == False:
                raise ValueError(
                    "The graph is not defined as node-weighted, but node weights are provided."
                )
            self.node_weighted = True
            self.nodes_weight = nodes_weight.astype(self.precision)
            self._check_nodes_weight_dim()
            self.nodes_num = int(nodes_weight.shape[0])
        
        if edges_weight is not None:
            if self.edge_weighted == False:
                raise ValueError(
                    "The graph is not defined as edge-weighted, but edge weights are provided."
                )
            self.edge_weighted = True
            self.edges_weight = edges_weight.astype(self.precision)
            self._check_edges_weight_dim()
            self.edges_num = int(edges_weight.shape[0])
        
        if edge_index is not None:
            self.edge_index = edge_index
            self._check_edges_index_dim()
            
            # Infer nodes_num and edges_num if not provided
            if self.nodes_num is None:
                self.nodes_num = int(np.max(edge_index) + 1)
            if self.edges_num is None:
                self.edges_num = int(edge_index.shape[1])
        
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()
            
        # Default Initialization if not provided
        if self.nodes_weight is None and self.nodes_num is not None:
            self.nodes_weight = np.ones(self.nodes_num, dtype=self.precision)
        
        if self.edges_weight is None and self.edges_num is not None:
            self.edges_weight = np.ones(self.edges_num, dtype=self.precision)
    
        # Make the graph symmetric
        if self.already_symmetric == False:
            self.make_symmetric()
            
        # Deal with self-loop
        if self.self_loop is None:
            self._deal_with_self_loop()
    
    def from_adj_matrix(
        self, 
        adj_matrix: np.ndarray, 
        nodes_weight: np.ndarray = None,
        edges_weight: np.ndarray = None,
    ):
        """Load graph data from an adjacency matrix."""
        # Check if the adjacency matrix is square
        if adj_matrix.ndim != 2:
            raise ValueError("Adjacency matrix should be a 2D array.")
        
        # Check if the adjacency matrix is all-ones
        if not np.all(np.isin(adj_matrix, [0, 1])):
            raise ValueError("Adjacency matrix should contain only 0s and 1s.")
        
        # Convert adjacency matrix to edge_index and edges_weight
        coo = scipy.sparse.coo_matrix(adj_matrix)
        edge_index = np.vstack((coo.row, coo.col)).astype(np.int32)
        
        # Use ``from_data``
        self.from_data(
            edge_index=edge_index,
            nodes_weight=nodes_weight,
            edges_weight=edges_weight
        )
    
    def from_adj_matrix_weighted(
        self, 
        adj_matrix_weighted: np.ndarray, 
        nodes_weight: np.ndarray = None,
    ):
        """Load graph data from an adjacency matrix."""
        # Check if the adjacency matrix is square
        if adj_matrix_weighted.ndim != 2:
            raise ValueError("Adjacency matrix should be a 2D array.")
        
        # Convert adjacency matrix to edge_index and edges_weight
        coo = scipy.sparse.coo_matrix(adj_matrix_weighted)
        edge_index = np.vstack((coo.row, coo.col)).astype(np.int32)
        if self.edge_weighted != False:
            edges_weight = edges_weight=coo.data.astype(self.precision)
        else:
            edges_weight = None
            
        # Use ``from_data``
        self.from_data(
            edge_index=edge_index,
            nodes_weight=nodes_weight,
            edges_weight=edges_weight
        )
    
    def to_adj_matrix(self, with_edge_weights: bool = False) -> np.ndarray:
        """Convert edge_index and edges_weight to adjacency matrix."""
        if with_edge_weights:
            if self.adj_matrix_weighted is None:
                self.adj_matrix_weighted = scipy.sparse.coo_matrix(
                    arg1=(
                        self.edges_weight, 
                        (self.edge_index[0], self.edge_index[1])
                    ), 
                    shape=(self.nodes_num, self.nodes_num)
                ).toarray().astype(self.precision)
            return self.adj_matrix_weighted
        else:
            if self.adj_matrix is None:
                self.adj_matrix = scipy.sparse.coo_matrix(
                    arg1=(
                        np.ones_like(self.edges_weight), 
                        (self.edge_index[0], self.edge_index[1])
                    ), 
                    shape=(self.nodes_num, self.nodes_num)
                ).toarray().astype(self.precision)
            return self.adj_matrix
    
    def from_gpickle_result(
        self, 
        gpickle_file_path: pathlib.Path = None,
        result_file_path: pathlib.Path = None, 
        ref: bool = False,
    ):
        """Load graph data from a gpickle file."""
        # Read graph data from .gpickle
        if gpickle_file_path is not None:
            with open(gpickle_file_path, "rb") as f:
                nx_graph: nx.Graph = pickle.load(f)

            # Use ``from_nx_graph``
            self.from_networkx(nx_graph)
            
        if result_file_path is not None:
            with open(result_file_path, "r") as f:
                sol = [int(_) for _ in f.read().splitlines()]
                
            # Use ``from_data``
            self.from_data(sol=np.array(sol, dtype=np.int32), ref=ref)
    
    def to_gpickle_result(
        self, 
        gpickle_file_path: pathlib.Path = None,
        result_file_path: pathlib.Path = None, 
    ):
        """Save graph data to a ``.gpickle`` or ``.result`` file."""
        # Save graph data to a .gpickle file
        if gpickle_file_path is not None:
            # Check file path
            check_file_path(gpickle_file_path)
            
            # Transfer to NetworkX graph
            nx_graph = self.to_networkx()
            
            # Save to .gpickle file
            with open(gpickle_file_path, "wb") as f:
                pickle.dump(nx_graph, f, pickle.HIGHEST_PROTOCOL)
        
        # Save graph data to a .result file
        if result_file_path is not None:
            # Check file path
            check_file_path(result_file_path)
            
            # Save to .result file
            with open(result_file_path, "w") as f:
                for node_label in self.sol:
                    f.write(f"{node_label}\n")
    
    def from_networkx(self, nx_graph: nx.Graph):
        """Load graph data from a NetworkX graph object."""
        # Extract nodes and edges information
        self.nodes_num = int(nx_graph.number_of_nodes())
        self.edges_num = int(nx_graph.number_of_edges())
        
        # Extract node weights if available
        nodes_weight = None
        if self.node_weighted != False and \
            all("weight" in nx_graph.nodes[n] for n in nx_graph.nodes):
            nodes_weight = np.array(
                [nx_graph.nodes[n]["weight"] for n in nx_graph.nodes], 
                dtype=self.precision
            )
            self.node_weighted = True
        else:
            nodes_weight = None
        
        # Extract edge weights if available
        edges_weight = None
        if self.edge_weighted != False and \
            all("weight" in nx_graph.edges[e] for e in nx_graph.edges):
            edges_weight = np.array(
                [nx_graph.edges[e]["weight"] for e in nx_graph.edges], 
                dtype=self.precision
            )
            self.edge_weighted = True
        else:
            edges_weight = None
        
        # Extract edge index
        edges = list(nx_graph.edges)
        edge_index = np.array(edges, dtype=np.int32).T
        
        # Use ``from_data``
        self.from_data(
            edge_index=edge_index,
            nodes_weight=nodes_weight,
            edges_weight=edges_weight
        )
    
    def to_networkx(self) -> nx.Graph:
        """Convert current graph data to a NetworkX graph object."""
        nx_graph = nx.Graph()
        
        # Add nodes with weights if available
        if self.nodes_weight is not None:
            for i in range(self.nodes_num):
                nx_graph.add_node(i, weight=self.nodes_weight[i])
        else:
            nx_graph.add_nodes_from(range(self.nodes_num))
        
        # Add edges with weights if available
        if self.edges_weight is not None:
            for i in range(self.edges_num):
                u = self.edge_index[0, i]
                v = self.edge_index[1, i]
                nx_graph.add_edge(u, v, weight=self.edges_weight[i])
        else:
            edges = list(zip(self.edge_index[0], self.edge_index[1]))
            nx_graph.add_edges_from(edges)
        
        return nx_graph

    def from_csr(
        self, 
        xadj: np.ndarray, 
        adjncy: np.ndarray,
        nodes_weight: np.ndarray = None, 
        edges_weight: np.ndarray = None
    ):
        """Load graph data from a CSR representation."""
        # Store CSR representation
        self.xadj = xadj
        self.adjncy = adjncy
        
        # Get edge_index from csr representation
        edge_index = [
            [src, dst] 
            for src in range(len(xadj) - 1) 
            for dst in adjncy[xadj[src]:xadj[src + 1]]
        ]
        edge_index = np.array(edge_index).T

        # Use ``from_data``
        self.from_data(
            edge_index=edge_index, 
            nodes_weight=nodes_weight, 
            edges_weight=edges_weight
        )

    def to_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert edge_index to CSR representation (xadj, adjncy)."""
        if self.xadj is None or self.adjncy is None:
            csr = scipy.sparse.csr_matrix(
                (np.ones(self.edges_num), (self.edge_index[0], self.edge_index[1])),
                shape=(self.nodes_num, self.nodes_num)
            )
            self.xadj = csr.indptr.astype(np.int32)
            self.adjncy = csr.indices.astype(np.int32)
        return self.xadj, self.adjncy

    def make_complement(self):
        """Convert the graph to its complement."""
        # Ensure the graph is not edge-weighted
        if self.edge_weighted:
            raise NotImplementedError("Complement of edge-weighted graphs is not supported.")
        
        # Get current adjacency matrix
        adj_matrix = self.to_adj_matrix()
        
        # Create complement adjacency matrix
        comp_adj = np.logical_not(adj_matrix).astype(int)
        
        # Invalidate cached structures
        self.edges_num = None
        self.edges_weight = None
        self._invalidate_cached_structures()
        
        # Use ``from_adj_matrix`` to update graph
        self.from_adj_matrix(adj_matrix=comp_adj)
    
    def make_symmetric(self):
        """Convert the graph to its symmetric."""
        # Step 1: Check if already symmetric
        adj_matrix_weighted = self.to_adj_matrix(with_edge_weights=True)
        
        # Step 2: Check for conflicting edges (both (i,j) and (j,i) exist)
        # Check if there are asymmetric edges where both directions exist
        asymmetric_mask = (adj_matrix_weighted != 0) & (adj_matrix_weighted.T != 0) \
            & (adj_matrix_weighted != adj_matrix_weighted.T)
        if np.any(asymmetric_mask):
            raise ValueError(
                "Cannot symmetrize graph: both (i,j) and (j,i) edges "
                "exist with different weights for some i!=j"
            )
        
        # Step 3: Perform symmetrization (both structural and weight)
        # Create symmetric adjacency matrix by taking maximum of original and transpose
        symmetric_adj = np.maximum(adj_matrix_weighted, adj_matrix_weighted.T)
        
        # Convert symmetric adjacency matrix back to edge_index and edges_weight
        coo = scipy.sparse.coo_matrix(symmetric_adj)
        edge_index = np.vstack((coo.row, coo.col)).astype(np.int32)
        if self.edge_weighted:
            edges_weight = coo.data.astype(self.precision)
        else:
            edges_weight = None
            
        # Invalidate cached structures
        self.edges_num = None
        self.edges_weight = None
        self._invalidate_cached_structures()
        
        # Using ``from_data``
        self.already_symmetric = True
        self.from_data(edge_index=edge_index, edges_weight=edges_weight)
    
    def add_self_loop(self):
        """Add self-loops to the graph."""
        # Remove existing self-loops
        self.remove_self_loop()
        
        # Create self-loop edges
        self_loops = np.arange(self.nodes_num, dtype=np.int32)
        self_loop_edges = np.vstack((self_loops, self_loops))
        self_loop_weights = np.ones(self.nodes_num, dtype=self.precision)
        
        # Combine with existing edges
        self.edge_index = np.hstack((self.edge_index, self_loop_edges))
        self.edges_weight = np.hstack((self.edges_weight, self_loop_weights))
        self.edges_num = int(self.edge_index.shape[1])
        
        # Invalidate cached structures
        self._invalidate_cached_structures()
        
    def remove_self_loop(self):
        """Remove self-loops from the graph."""
        # Identify non-self-loop edges
        mask = self.edge_index[0] != self.edge_index[1]
        self.edge_index = self.edge_index[:, mask]
        self.edges_weight = self.edges_weight[mask]
        self.edges_num = int(self.edge_index.shape[1])
        
        # Invalidate cached structures
        self._invalidate_cached_structures()
        

# NetworkX Layout
SUPPORT_POS_TYPE_DICT = {
    "bipartite_layout": nx.bipartite_layout,
    "circular_layout": nx.circular_layout,
    "kamada_kawai_layout": nx.kamada_kawai_layout,
    "random_layout": nx.random_layout,
    "rescale_layout": nx.rescale_layout,
    "rescale_layout_dict": nx.rescale_layout_dict,
    "shell_layout": nx.shell_layout,
    "spring_layout": nx.spring_layout,
    "spectral_layout": nx.spectral_layout,
    "planar_layout": nx.planar_layout,
    "fruchterman_reingold_layout": nx.fruchterman_reingold_layout,
    "spiral_layout": nx.spiral_layout,
    "multipartite_layout": nx.multipartite_layout,
}


# Supported Pos Types
SUPPORT_POS_TYPE = [
    "bipartite_layout",
    "circular_layout",
    "kamada_kawai_layout",
    "random_layout",
    "rescale_layout",
    "rescale_layout_dict",
    "shell_layout",
    "spring_layout",
    "spectral_layout",
    "planar_layout",
    "fruchterman_reingold_layout",
    "spiral_layout",
    "multipartite_layout",
]


# Get Position Layer
def get_pos_layer(pos_type: str):
    if pos_type not in SUPPORT_POS_TYPE:
        raise ValueError(f"unvalid pos type, only supports {SUPPORT_POS_TYPE}")
    return SUPPORT_POS_TYPE_DICT[pos_type]