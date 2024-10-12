import copy
import numpy as np
import scipy.sparse
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
    
    def to_matrix(self):
        if self.adj_matrix is None:
            self.adj_matrix = np_sparse_to_dense(
                nodes_num=self.nodes_num,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr
            )
        return self.adj_matrix
    
    
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