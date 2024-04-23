import numpy as np
from typing import Union


def to_numpy(x: Union[np.ndarray, list]):
    if type(x) == list:
        return np.array(x)
    elif type(x) == np.ndarray:
        return x
    else:
        raise NotImplementedError()