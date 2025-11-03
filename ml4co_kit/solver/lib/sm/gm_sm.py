r"""
SM Algorithm for GM
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
from ml4co_kit.task.graph.gm import GMTask


def gm_sm(
    task_data: GMTask,
    x0: np.ndarray,
    max_iter: int = 50,
):
    K = task_data.aff_matrix
    if K is None:
        raise ValueError("Affinity matrix need to be built.")
    n1 = task_data.graphs[0].nodes_num
    n2 = task_data.graphs[1].nodes_num
    n1, n2, n1n2, v0 = _check_and_init_gm(K, n1, n2)
    v = vlast = v0
    for i in range(max_iter):
        v = np.matmul(K, v)
        n = np.linalg.norm(v, ord=2)
        v = v / n
        if np.linalg.norm(v-vlast, ord='fro') < 1e-8:
            print("这一轮已经收敛")
            break
        vlast = v
        if i == max_iter-1:
            print("这一轮未收敛")
    
    x = v.reshape((n2, n1)).T
    
    return x
   
    
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