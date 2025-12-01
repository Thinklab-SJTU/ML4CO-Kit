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


import pickle
import pathlib
import numpy as np
import scipy.sparse
import networkx as nx
from typing import Union
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path


class Graph:
    def __init__(
        self,
        nodes_num: int = None,
        edges_num: int = None, 
        node_feature: np.ndarray = None,
        node_feat_dim: int = None,
        edge_feature: np.ndarray = None,
        edge_feat_dim: int = None,
        edge_index: np.ndarray = None,
        precision: Union[np.float32, np.float64] = np.float32
    ):
            
        # Initialize Attributes (basic)   
        self.nodes_num = nodes_num
        self.edges_num = edges_num
        self.node_feature = node_feature
        self.node_feat_dim =  node_feat_dim
        self.edge_feature = edge_feature
        self.edge_feat_dim =  edge_feat_dim
        self.edge_index =  edge_index                # [Method 1] Edge Index in shape (2, num_edges)
        self.precision = precision
            
        # Symmetric    
        self.already_symmetric = False
        
        #self-loop
        self.self_loop = None
            
        # Initialize Attributes (other structure)    
        self.adj_matrix: np.ndarray = None           # [Method 2] Adjacency Matrix
        self.featured_adj_matrix: np.ndarray = None  # [Method 3] Adjacency Matrix with edges_weight
        self.xadj: np.ndarray = None                 # [Method 4] Compressed Sparse Row (CSR) representation
        self.adjncy: np.ndarray = None               # [Method 4] Compressed Sparse Row (CSR) representation   
        
    def _check_nodes_feature(self):
        """Ensure node feature is a 2D array with correct feature dimension."""
        if self.node_feature.ndim != 2:
            raise ValueError("Node feature should be a 2D array with shape (num_nodes,).")
        
        if self.node_feat_dim is not None and self.node_feature.shape[1] != self.node_feat_dim:
            raise ValueError("Node feature dimension mismatch") 
    
    def _check_edges_feature(self):
        """Ensure edge feature is a 2D array with correct feature dimension ."""
        if self.edge_feature.ndim != 2:
            raise ValueError("Edge feature should be a 2D array with shape (num_edges,).")   
        
        if self.edge_feat_dim is not None and self.edge_feature.shape[1] != self.edge_feat_dim:
            raise ValueError("Edge feature dimension mismatch")         
            
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
        
    def _invalidate_cached_structures(self):
        """Invalidate cached structures."""
        self.adj_matrix = None
        self.featured_adj_matrix = None
        self.xadj = None
        self.adjncy = None
           
    def from_data(
        self,
        edge_index: np.ndarray = None,
        node_feature: np.ndarray = None,
        edge_feature: np.ndarray = None,
        self_loop: bool = False,
    ):      
                
        if node_feature is not None:
            self.node_feature = node_feature.astype(self.precision)
            self._check_nodes_feature()
            self.node_feat_dim = self.node_feature.shape[1]
            self.nodes_num = int(node_feature.shape[0])
                             
              
        if edge_feature is not None:
            self.edge_feature = edge_feature.astype(self.precision)
            self._check_edges_feature()
            self.edge_feat_dim = self.edge_feature.shape[1]
            self.edges_num = int(edge_feature.shape[0])      
            
        if edge_index is not None:
            self.edge_index = edge_index
            self._check_edges_index_dim()  
                
            # Infer nodes_num and edges_num if not provided
            if self.nodes_num is None:
               self.nodes_num = int(np.max(edge_index) + 1)
            if self.edges_num is None:
               self.edges_num = int(edge_index.shape[1])
               
        # Default Initialization if not provided
        if self.node_feature is None and self.nodes_num is not None:
            if self.node_feat_dim is None:
                raise ValueError("Both node feature and its dimension are not provided")
            self.node_feature = np.ones((self.nodes_num, self.node_feat_dim), dtype=self.precision)
        
        if self.edge_feature is None and self.edges_num is not None:
            if self.edge_feat_dim is None:
                raise ValueError("Both edge feature and its dimension are not provided")
            self.edge_feature = np.ones((self.edges_num, self.edge_feat_dim), dtype=self.precision)
         
        self.self_loop = self_loop   
                    
        # Make the graph symmetric
        if self.already_symmetric == False:
            self.make_symmetric()
            
                         
    def from_adj_matrix(
        self, 
        adj_matrix: np.ndarray, 
        node_feature: np.ndarray = None,
        edge_feature: np.ndarray = None,
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
            node_feature=node_feature,
            edge_feature=edge_feature
        )
                  
    def from_featured_adj_matrix(
        self, 
        featured_adj_matrix: np.ndarray, 
        node_feature: np.ndarray = None,
    ):
        """Load graph data from an adjacency matrix."""
        # Check if the adjacency matrix is square
        if featured_adj_matrix.ndim != 3:
            raise ValueError("Adjacency matrix should be a 3D array.")
        
        # Convert adjacency matrix to edge_index and edge_feature
        adj_matrix = np.any(featured_adj_matrix != 0, axis =2)
        coo = scipy.sparse.coo_matrix(adj_matrix)
        edge_index = np.vstack((coo.row, coo.col)).astype(np.int32)
        
        edge_feature =featured_adj_matrix[coo.row, coo.col] 
            
        # Use ``from_data``
        self.from_data(
            edge_index=edge_index,
            node_feature=node_feature,
            edge_feature=edge_feature
        )
    
    def to_adj_matrix(self, with_edge_features: bool = False) -> np.ndarray:
        """Convert edge_index and edges_feature to adjacency matrix."""
        if with_edge_features:
            if self.featured_adj_matrix is None:
               self.featured_adj_matrix = np.zeros((self.nodes_num, self.nodes_num, self.edges_feature.shape[1])).astype(self.precision)
               self.featured_adj_matrix[self.edge_index[0], self.edge_index[1], :] = self.edges_feature
            return self.featured_adj_matrix
        else:
            if self.adj_matrix is None:
                self.adj_matrix = scipy.sparse.coo_matrix(
                    arg1=(
                        np.ones(self.edges_num, dtype=self.precision), 
                        (self.edge_index[0], self.edge_index[1])
                    ), 
                    shape=(self.nodes_num, self.nodes_num)
                ).toarray().astype(self.precision)
            return self.adj_matrix
          
    def from_networkx(self, nx_graph: nx.Graph):
        """Load graph data from a NetworkX graph object."""
        # Extract nodes and edges information
        self.nodes_num = int(nx_graph.number_of_nodes())
        self.edges_num = int(nx_graph.number_of_edges())
        
        # Extract node feature if available
        node_feature = None
        if  nx_graph.number_of_nodes() > 0 and all("feature" in nx_graph.nodes[n] for n in nx_graph.nodes):
            node_feature = np.array(
                [nx_graph.nodes[n]["feature"] for n in nx_graph.nodes], 
                dtype=self.precision
            )
        else:
            node_feature = None
        
        # Extract edge feature if available
        edge_feature = None
        if  nx_graph.number_of_edges() > 0 and all("feature" in nx_graph.edges[e] for e in nx_graph.edges):
            edge_feature = np.array(
                [nx_graph.edges[e]["feature"] for e in nx_graph.edges], 
                dtype=self.precision
            )
        else:
            edge_feature = None
        
        # Extract edge index
        edge_index = None
        if nx_graph.number_of_edges() > 0:
            edges = list(nx_graph.edges)
            edge_index = np.array(edges, dtype=np.int32).T
        else:
            edge_index = None
        
        # change undirected graph to directed graph
        if edge_index is not None and edge_index.size != 0:
            reversed_edges = edge_index[[1, 0], :]
            edge_index = np.concatenate([edge_index, reversed_edges], axis=1)

        if edge_feature is not None and edge_feature.size != 0:
            edge_feature = np.concatenate([edge_feature, edge_feature], axis=0)
        # Use ``from_data``
        self.from_data(
            edge_index=edge_index,
            node_feature=node_feature,
            edge_feature=edge_feature
        )        
        
    def to_networkx(self) -> nx.Graph:
        """Convert current graph to NetworkX graph object."""
        nx_graph = nx.Graph()
        
        # Add nodes with feature
        if self.node_feature is not None:
            for i in range(self.nodes_num):
                nx_graph.add_node(i, feature=self.node_feature[i])
        else:
            for i in range(self.nodes_num):
                nx_graph.add_node(i, feature=np.ones(self.node_feat_dim, dtype=self.precision))
        
        # Add edges with feature 
        if self.edge_feature is not None:
            for i in range(self.edges_num):
                u = self.edge_index[0, i]
                v = self.edge_index[1, i]
                nx_graph.add_edge(u, v, feature=self.edge_feature[i])
        else:
            for i in range(self.edges_num):
                u = self.edge_index[0, i]
                v = self.edge_index[1, i]
                nx_graph.add_edge(u, v, feature=np.ones(self.edge_feat_dim, dtype=self.precision))
        
        return nx_graph
    
    def make_symmetric(self):
        """Convert the graph to its symmetric."""
        # Step 1: construct feature matrix
        n = self.nodes_num
        edge_feat_dim = self.edge_feature.shape[1]
        
        adj_matrix = np.zeros((n, n, edge_feat_dim), dtype=self.precision)
        rows, cols = self.edge_index
        adj_matrix[rows, cols, :] = self.edge_feature
        
        # Step 2: Check for conflicting edges (both (i,j) and (j,i) exist)
        # Check if there are asymmetric edges where both directions exist
        mask_exist = np.any(adj_matrix != 0, axis=2)
        mask_feat = np.any(adj_matrix != adj_matrix.transpose(1,0,2), axis=2)
        asymmetric_mask = mask_exist & mask_exist.T & mask_feat
        if np.any(asymmetric_mask):
            raise ValueError(
                "Cannot symmetrize graph: both (i,j) and (j,i) edges "
                "exist with different features for some i!=j"
            )
        
        # Step 3: Perform symmetrization (both structural and feature)
        mask_single = mask_exist & (~mask_exist.T)
        row_idx, col_idx = np.nonzero(mask_single)
        adj_matrix[col_idx, row_idx, :] = adj_matrix[row_idx, col_idx, :]
        
        
        # Convert symmetric adjacency matrix back to edge_index and edges_weight
        row_idx, col_idx = np.nonzero(np.any(adj_matrix != 0, axis=2))
        edge_index = np.vstack((row_idx, col_idx)).astype(np.int32)
        edge_feature = adj_matrix[row_idx, col_idx, :]
        
        # Invalidate cached structures
        self.edges_num = None
        self.edge_feature = None
        self._invalidate_cached_structures()
        
        # Using ``from_data``
        self.already_symmetric = True
        self.from_data(edge_index=edge_index, edge_feature=edge_feature)
        
    def remove_self_loop(self):
        mask = self.edge_index[0] != self.edge_index[1]
        self.edge_index = self.edge_index[:, mask]
        self.edges_num = int(self.edge_index.shape[1])
        self.edge_feature = self.edge_feature[mask]
        self._invalidate_cached_structures()
                
    def add_self_loop(self):
        self.remove_self_loop()
        self_loops = np.arange(self.nodes_num, dtype=np.int32)
        self_loop_edges = np.vstack((self_loops, self_loops))
        self_loop_edges_feature = np.ones((self.nodes_num, self.edge_feature.shape[1]), dtype = self.precision)
                
        self.edge_index = np.hstack((self.edge_index, self_loop_edges))
        self.edge_feature = np.vstack((self.edge_feature, self_loop_edges_feature))
        self.edges_num = int(self.edge_index.shape[1])
        self._invalidate_cached_structures()
       
class GraphSetTaskBase(TaskBase):                   
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool,
        graphs: list[Graph] = None,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super().__init__(
            task_type=task_type, 
            minimize=minimize, 
            precision=precision
            ) 
        self.graphs: list[Graph] = graphs if graphs is not None else [] 
        self.graphs_num: int = len(self.graphs)
        
    def add_graph(self, graph: Graph):
        if not isinstance(graph, Graph): 
            raise TypeError("Graph must be an instance of Graph") 
        self.graphs.append(graph) 
        self.graphs_num = len(self.graphs) 
        
    def remove_graph(self, idx: int): 
        if idx < 0 or idx >= len(self.graphs): 
            raise IndexError("Graph index out of range.") 
        del self.graphs[idx] 
        self.graphs_num = len(self.graphs) 
        
    def clear_graphs(self):
        if self.graphs is not None: 
           self.graphs.clear() 
        self.graphs_num = 0 
        self.ref_sol = None
        self.sol = None
        
    def get_graph(self, idx: int) -> Graph: 
        if idx < 0 or idx >= len(self.graphs): 
            raise IndexError("Graph index out of range.")
        return self.graphs[idx] 
    
    def _deal_with_self_loop(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _check_sol_dim(self):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def _check_ref_sol_dim(self):
       raise NotImplementedError("Subclasses should implement this method.")        
   
    def from_data( 
        self,
        graphs: list[Graph] = None,
        sol: np.ndarray = None,
        ref: bool = False,
        ):
        if graphs is not None:
            self.clear_graphs() 
            self.graphs = graphs  
            self.graphs_num = len(self.graphs)
            self._deal_with_self_loop()
        
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()
          
    def from_gpickle_result(
        self, 
        gpickle_file_path: pathlib.Path = None,
        result_file_path: pathlib.Path = None, 
        ref: bool = False,
    ):
        """Load graphs data from a gpickle file."""
        # Read graph data from .gpickle
        if gpickle_file_path is not None:
            with open(gpickle_file_path, "rb") as f:
                nx_graphs: list[nx.Graph] = pickle.load(f)

            # Use ``from_nx_graph``
            graphs_list = []
            for nx_graph in nx_graphs:
                g = Graph()
                g.from_networkx(nx_graph)
                graphs_list.append(g)
            self.graphs = graphs_list
            self.graphs_num = len(self.graphs)
            
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
            
            # Transfer to NetworkX graph list
            nx_graphs = [g.to_networkx for g in self.graphs]
            
            # Save to .gpickle file
            with open(gpickle_file_path, "wb") as f:
                pickle.dump(nx_graphs, f, pickle.HIGHEST_PROTOCOL)
        
        # Save graph data to a .result file
        if result_file_path is not None:
            # Check file path
            check_file_path(result_file_path)
            
            # Save to .result file
            with open(result_file_path, "w") as f:
                for node_label in self.sol:
                    f.write(f"{node_label}\n")

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
