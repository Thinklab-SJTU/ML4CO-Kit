import torch
import numpy as np
from torch import Tensor
from ml4co_kit.learning.utils import to_numpy, to_tensor, check_dim


def tsp_2opt_local_search(
    init_tours: np.ndarray,
    points: np.ndarray, 
    max_iterations_2opt: int = 5000, 
    device: str = "cpu"
) -> np.ndarray:    
    # init iterator
    iterator = 0 
    
    # local search
    with torch.inference_mode():
        # preparation
        init_tours = np.expand_dims(init_tours, axis=0)
        tours: Tensor = to_tensor(init_tours).to(device)
        points: Tensor = to_tensor(points).to(device)

        # start 2opt
        min_change = -1.0
        batch_size = 1
        while min_change < 0.0:
            # points
            points_i = points[tours[:, :-1].reshape(-1)]
            points_i = points_i.reshape(batch_size, -1, 1, 2)
            points_j = points[tours[:, :-1].reshape(-1)]
            points_j = points_j.reshape(batch_size, 1, -1, 2)
            points_i_plus_1 = points[tours[:, 1:].reshape(-1)]
            points_i_plus_1 = points_i_plus_1.reshape(batch_size, -1, 1, 2)
            points_j_plus_1 = points[tours[:, 1:].reshape(-1)]
            points_j_plus_1 = points_j_plus_1.reshape(batch_size, 1, -1, 2)
            
            # distance matrix
            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(
                torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1)
            )
            A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
            A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))
            
            # change
            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            # min change
            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, len(points))

            # check min change
            if min_change < -1e-6:
                for i in range(batch_size):
                    tours[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(
                        tours[i, min_i[i] + 1:min_j[i] + 1], dims=(0,)
                    )
                iterator += 1
            else:
                break
                
            # check iteration
            if iterator >= max_iterations_2opt:
                break
    
    # return
    return to_numpy(tours[0])