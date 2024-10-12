import torch
import numpy as np
from torch import Tensor
from typing import Union, Tuple
from sklearn.neighbors import KDTree


#####################################################
#                    Check Utils                    #
#####################################################

def check_dim(array: Union[np.ndarray, Tensor], dim: int):
    if isinstance(array, np.ndarray):
        if array.ndim != dim:
            raise ValueError("Dimension mismatch!")
    if isinstance(array, Tensor):
        if array.ndim != dim:
            raise ValueError("Dimension mismatch!")
        

#####################################################
#                     Type Utils                    #
#####################################################

def to_numpy(
    x: Union[np.ndarray, Tensor, list]
) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, list):
        return np.array(x)
    
    
def to_tensor(
    x: Union[np.ndarray, Tensor, list]
) -> Tensor:
    if isinstance(x, Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list):
        return Tensor(x)
    

#####################################################
#                    Graph Utils                    #
#####################################################

def sparse_points(
    points: Union[np.ndarray, Tensor], sparse_factor: int, device: str = "cpu"
) -> Tuple[Tensor, Tensor]:
    points = to_numpy(points)
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)
    
    edge_index = list()
    for idx in range(points.shape[0]):
        kdt = KDTree(points[idx], leaf_size=30, metric='euclidean')
        _, idx_knn = kdt.query(points[idx], k=sparse_factor, return_distance=True)
        _edge_index_0 = torch.arange(points[idx].shape[0])
        _edge_index_0 = _edge_index_0.reshape((-1, 1))
        _edge_index_0 = _edge_index_0.repeat(1, sparse_factor)
        _edge_index_0 = _edge_index_0.reshape(-1)
        _edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
        _edge_index = torch.stack([_edge_index_0, _edge_index_1], dim=0)
        edge_index.append(_edge_index.unsqueeze(dim=0))
    edge_index = torch.cat(edge_index, dim=0).to(device)

    points = torch.from_numpy(points).to(device)
    return points, edge_index


def points_to_distmat(points: Tensor, edge_index: Tensor = None) -> Tensor:
    device = points.device
    if edge_index is None:
        if points.ndim == 2:
            distmat = torch.cdist(points, points)
        else:
            distmat = torch.zeros(size=(points.shape[0], points.shape[1], points.shape[1]))
            for i, matrix in enumerate(points):
                distmat[i] = torch.cdist(matrix, matrix)
    else:     
        if points.ndim == 2:
            x = edge_index[0]
            y = edge_index[1]
            points_x = points[x]
            points_y = points[y]
            distmat = torch.norm(points_x - points_y, dim=1)
        else:
            matrix_list = list()
            for i in range(points.shape[0]):
                x = edge_index[i][0]
                y = edge_index[i][1]
                points_x = points[i][x]
                points_y = points[i][y]
                matrix_list.append(torch.norm(points_x - points_y, dim=1))
            distmat = torch.stack(matrix_list)
    return distmat.to(device)
