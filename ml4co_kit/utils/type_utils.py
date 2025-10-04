import torch
import numpy as np
from torch import Tensor
from typing import Union


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