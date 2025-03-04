import copy
import pickle
import numpy as np
import scipy.sparse
import networkx as nx
from enum import Enum
from typing import Union, Tuple


class Dense2SparseType(str, Enum):
    DISTANCE = "distance"
    ZERO_ONE = "zero-one"


class GraphData(object):
    def __init__(self):
        self.x = None
        self.edge_index: np.ndarray = None
        self.edge_attr: np.ndarray = None
        self.adj_matrix: np.ndarray = None
        self.nodes_num: np.ndarray = None
        
    def from_adj_martix(
        self, 
        adj_matrix: np.ndarray,
        max_or_min: str = "min",
        zero_or_one: str = "one",
        type: Dense2SparseType = Dense2SparseType.DISTANCE, 
        sparse_factor: int = None,
        self_loop: bool = None,
    ):
        nodes_num, edge_index, edge_attr = np_dense_to_sparse(
            adj_matrix=adj_matrix, max_or_min=max_or_min,
            zero_or_one=zero_or_one, type=type, 
            sparse_factor=sparse_factor, self_loop=self_loop
        )
        self.nodes_num = nodes_num
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
    def from_graph_data(
        self, 
        x: np.ndarray = None, 
        edge_index: np.ndarray = None, 
        edge_attr: np.ndarray = None
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        if x is not None:
            self.nodes_num = len(x)
    
    def from_gpickle(
        self, file_path: str, self_loop: bool = True
    ):
        # check file format
        if not file_path.endswith(".gpickle"):
            raise ValueError("Invalid file format. Expected a ``.gpickle`` file.")
        
        # read graph data from .gpickle
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
        graph: nx.Graph

        # use ``from_nx_graph``
        self.from_nx_graph(graph, self_loop=self_loop)        
    
    def from_nx_graph(self, nx_graph: nx.Graph, self_loop: bool = True):
        # nodes num
        self.nodes_num = nx_graph.number_of_nodes()
        
        # edges
        edges = np.array(nx_graph.edges, dtype=np.int64)
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        self.self_loop = self_loop
        if self.self_loop:
            self_loop: np.ndarray = np.arange(self.nodes_num)
            self_loop = self_loop.reshape(-1, 1).repeat(2, axis=1)
            edges = np.concatenate([self_loop, edges], axis=0)
        edges = edges.T

        # use ``from_data``
        self.from_data(edge_index=edges)
    
    def from_result(self, file_path: str, ref: bool = False):
        # check file format
        if not file_path.endswith(".result"):
            raise ValueError("Invalid file format. Expected a ``.result`` file.")
        
        # read solution from file
        with open(file_path, "r") as f:
            nodes_label = [int(_) for _ in f.read().splitlines()]
        nodes_label = np.array(nodes_label, dtype=np.int64)
        
        # use ``from_data``
        self.from_data(nodes_label=nodes_label, ref=ref)  
        
    def from_data(
        self, 
        edge_index: np.ndarray = None, 
        nodes_label: np.ndarray = None,
        ref: bool = False
    ):
        if edge_index is not None:
            self.edge_index = edge_index
        if nodes_label is not None:
            if ref:
                self.ref_nodes_label = nodes_label
            else:
                self.nodes_label = nodes_label
    
    def to_matrix(self):
        if self.adj_matrix is None:
            self.adj_matrix = np_sparse_to_dense(
                nodes_num=self.nodes_num,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr
            )
        return self.adj_matrix
    
    def to_networkx(self) -> nx.Graph:
        """
        Converts the GraphData instance to a networkx Graph.
        """
        nx_graph = nx.Graph()
        
        # Add nodes
        if self.nodes_num is not None:
            nx_graph.add_nodes_from(range(self.nodes_num))
        
        # Add edges with attributes
        if self.edge_index is not None:
            edges = self.edge_index.T  # Transpose to get pairs of edges
            if self.edge_attr is not None:
                edge_data = zip(edges, self.edge_attr)  
            else:
                edge_data = zip(edges, [1] * edges.shape[0])
            
            for (u, v), weight in edge_data:
                nx_graph.add_edge(u, v, weight=weight)
        
        return nx_graph
    
    def to_complement(self):
        """
        Converts the current graph to its complement by reversing the edge relationships
        between nodes. Self-loop configurations remain unchanged.
        """
        if self.adj_matrix is None:
            # If no adjacency matrix is present, generate it first
            self.to_matrix()
        
        # Generate the complement adjacency matrix by inverting 0s and 1s
        complement_adj_matrix = np.logical_not(self.adj_matrix).astype(int)
        
        # Preserve the diagonal values for self-loops (change to 0 to remove self-loops)
        np.fill_diagonal(complement_adj_matrix, self.adj_matrix.diagonal())
        
        # Update the current graph data to reflect the complement graph
        self.from_adj_martix(complement_adj_matrix)
    
    def remove_self_loop(self):
        """
        Removes self-loops from the edge data by filtering out edges where source and target nodes are the same.
        Updates `edge_index` and `edge_attr` accordingly.
        """
        if self.edge_index is not None:
            mask = self.edge_index[0] != self.edge_index[1]  # Mask to filter non-self-loops
            self.edge_index = self.edge_index[:, mask]
            if self.edge_attr is not None:
                self.edge_attr = self.edge_attr[mask]

    def add_self_loop(self, loop_weight: Union[int, float] = 1):
        """
        Adds self-loops to the graph, updating `edge_index` and `edge_attr` accordingly.
        Args:
            loop_weight (int, float): The weight to assign to each self-loop edge.
        """
        if np.any(np.all(self.edge_index.T == np.array([0,0]), axis=1)):
            return
        
        if self.nodes_num is None:
            raise ValueError("`nodes_num` must be defined to add self-loops.")

        self_loops = np.arange(self.nodes_num)  # Diagonal edges for each node
        loop_edges = np.stack((self_loops, self_loops), axis=0)
        
        # Append self-loops to edge_index and edge_attr
        self.edge_index = (
            np.hstack((self.edge_index, loop_edges)) 
            if self.edge_index is not None 
            else loop_edges
        )
        
        if self.edge_attr is not None:
            loop_attrs = np.full((self.nodes_num,), loop_weight)
            self.edge_attr = np.hstack((self.edge_attr, loop_attrs))
        else:
            edges_num = self.edge_index.shape[1]
            self.edge_attr = np.full((edges_num,), loop_weight)
        
    def check_edge_attr(self):
        if self.edge_attr is None:
            edge_nums = self.edge_index.shape[1]
            self.edge_attr = np.ones(shape=(edge_nums,))
        
        
def np_sparse_to_dense(
    nodes_num: Union[int, list, tuple], 
    edge_index: np.ndarray, 
    edge_attr: np.ndarray = None, 
) -> np.ndarray:
    """
    edge_index: (2, E)
    edge_attr: (E,) if is None, apply ``All-Ones`` 
    """
    # edge attr
    if edge_attr is None:
        edge_nums = edge_index.shape[1]
        edge_attr = np.ones(shape=(edge_nums,))
    
    # shape
    if isinstance(nodes_num, int):
        shape = (nodes_num, nodes_num)
    else:
        shape = (nodes_num[0], nodes_num[1])
        
    # sparse to dense
    adj_matrix = scipy.sparse.coo_matrix(
        arg1=(edge_attr, (edge_index[0], edge_index[1])), shape=shape
    ).toarray()
    
    # return
    return adj_matrix


def np_dense_to_sparse(
    adj_matrix: np.ndarray,
    max_or_min: str = "min",
    zero_or_one: str = "one",
    type: Dense2SparseType = Dense2SparseType.DISTANCE, 
    sparse_factor: int = None,
    self_loop: bool = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    # check dimension
    if adj_matrix.ndim != 2:
        raise ValueError("Dimension of input array must be 2!")
    
    # nodes num
    nodes_num = adj_matrix.shape[0]
    
    # dense to sparse (distance)
    if type == Dense2SparseType.DISTANCE:
        if sparse_factor is None:
            raise ValueError(
                "``sparse_factor`` can not be None if type is ``distance``"
            )
        
        # max or min
        if max_or_min == "max":
            adj_matrix = -adj_matrix
        
        # KNN  
        new_adj_matrix = copy.deepcopy(adj_matrix)
        min_value, max_value = adj_matrix.min(), adj_matrix.max()
        if self_loop == True:
            new_adj_matrix[range(nodes_num), range(nodes_num)] = min_value - 1
        elif self_loop == False:
            new_adj_matrix[range(nodes_num), range(nodes_num)] = max_value + 1
        elif self_loop is None:
            pass
        idx_knn = np.argsort(new_adj_matrix, axis=1)[:, :sparse_factor]

        # edge_index && edge_attr
        edge_index_0 = np.arange(nodes_num).reshape((-1, 1))
        edge_index_0 = edge_index_0.repeat(sparse_factor, 1).reshape(-1)
        edge_index_1 = idx_knn.reshape(-1)
        edge_index = np.stack([edge_index_0, edge_index_1], axis=0)
        
        # edge_attr
        edge_attr = adj_matrix[edge_index_0, idx_knn.reshape(-1)]
        if max_or_min == "max":
            edge_attr = -edge_attr
            
    # dense to sparse (zero-one)
    elif type == Dense2SparseType.ZERO_ONE:
        # check zero or one matrix
        if not np.all(np.in1d(adj_matrix, [0, 1])):
            raise ValueError(
                "When the type is ``zero-one``, the elements in the matrix must be either 0 or 1."
            )
        # zero or one
        if zero_or_one == "zero":
            adj_matrix = 1 - adj_matrix
        
        # self loop
        if self_loop == True:
            adj_matrix[range(nodes_num), range(nodes_num)] = 1
        elif self_loop == False:
            adj_matrix[range(nodes_num), range(nodes_num)] = 0
        else:
            pass
                        
        # find all '1' elements
        edge_index_0, edge_index_1 = np.where(adj_matrix == 1) 
        edge_index = np.stack([edge_index_0, edge_index_1], axis=0)
        
        # edge_attr
        edges_num = edge_index.shape[1]
        if zero_or_one == "zero":
            edge_attr = np.zeros(shape=(edges_num,))
        else:
            edge_attr = np.ones(shape=(edges_num,))

    # return
    return nodes_num, edge_index, edge_attr