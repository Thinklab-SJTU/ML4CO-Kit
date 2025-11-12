r"""
RRWM Algorithm for GM
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

import os
import tempfile
import numpy as np
from ml4co_kit.task.graphset.gm import GMTask
from ml4co_kit.task.graphset.base import sinkhorn, hungarian

def gm_rrwm(
    task_data: GMTask,
    x0: np.ndarray,
    max_iter: int = 50,
    sk_iter: int = 20,
    alpha: float = 0.2,
    beta: float = 30.0,
):
    if task_data.aff_matrix is None:
        task_data.build_aff_mat()
     
    K = task_data.aff_matrix
    n1 = task_data.graphs[0].nodes_num
    n2 = task_data.graphs[1].nodes_num
    n1, n2, n1n2, v0 = _check_and_init_gm(K, n1, n2)
    
    d = K.sum(axis=1, keepdims=True)
    dmax = d.max(axis=0, keepdims=True)
    K = K / (dmax + d.min() * 1e-5)
    v = v0
    for i in range(max_iter):
        # random walk
        v = np.matmul(K, v)
        last_v = v.copy()
        n = np.linalg.norm(v, ord=1, axis=0, keepdims=True)
        v = v / n
        
        # reweighted jump
        s = v.reshape(n2, n1).transpose((1,0))
        s = beta * s / np.amax(s, axis=(0,1), keepdims=True)
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter).transpose((1,0)).reshape((n1n2, 1)) + \
            (1-alpha) * v
        n = np.linalg.norm(v, ord=1, axis=0, keepdims=True)
        v = v / n
        
        if np.linalg.norm(v - last_v) < 1e-5:
            break
    
    v = v.reshape((n2, n1)).transpose((1, 0))
    v = hungarian(v)
    task_data.from_data(sol=v.reshape((n1n2,)), ref=False)
    return v

def _check_and_init_gm(K: np.ndarray, n1: int = None, n2: int = None, x0: np.ndarray = None):
    n1n2 = K.shape[0]
 
    if n1 is None and n2 is None:
        raise ValueError('Neither n1 or n2 is given.')
    if n1 is None:
        if n1n2 % n2 == 0:
            n1 = n1n2 / n2
        else:
            raise ValueError("The input size of K does not match with n2!")
    if n2 is None:
        if n1n2 % n1 == 0:
            n2 = n1n2 / n1
        else:
            raise ValueError("The input size of K does not match with n1!")
    if not n1 * n2 == n1n2:
        raise ValueError('the input size of K does not match with n1 * n2!')

    # initialize x0 (also v0)
    if x0 is None:
        x0 = np.zeros((n1, n2), dtype=K.dtype)
        x0[:] = 1. / (n1 * n2)
    v0 = x0.transpose((1, 0)).reshape((n1n2, 1))

    return n1, n2, n1n2, v0