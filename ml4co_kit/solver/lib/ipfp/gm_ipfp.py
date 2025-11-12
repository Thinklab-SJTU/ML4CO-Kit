r"""
IPFP Algorithm for GM
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
from ml4co_kit.task.graphset.gm import GMTask
from ml4co_kit.task.graphset.base import hungarian

def gm_ipfp(task_data: GMTask, x0: np.ndarray = None, max_iter: int = 50) -> np.ndarray:
    """Single-graph version of IPFP algorithm"""
    if task_data.aff_matrix is None:
        task_data.build_aff_mat()
     
    K = task_data.aff_matrix
    n1 = task_data.graphs[0].nodes_num
    n2 = task_data.graphs[1].nodes_num
    n1, n2, n1n2, v0 = _check_and_init_gm(K, n1, n2, x0)
    v = v0
    last_v = v
    best_v = v
    best_obj = -1
    
    def comp_obj_score(v1, K, v2):
        return v1.T @ K @ v2 

    for i in range(max_iter):
        cost = (K @ v).reshape((n2, n1)).T
        gm=GMTask()  
        binary_sol = hungarian(s=cost, n1=n1, n2=n2) 
        binary_v = binary_sol.T.reshape((n1*n2, 1))
        d = binary_v - v
        alpha = comp_obj_score(v, K, d)
        beta = comp_obj_score(d, K, d)
        if np.abs(beta) < 1e-10:
            t0=0
        else:
            t0 = -alpha / beta
            
        if beta >= 0 or t0 >= 1:
            v = binary_v
        else:
            v = v + t0 * d
            
        last_v_obj = comp_obj_score(last_v, K, last_v)    
        current_obj = comp_obj_score(binary_v, K, binary_v)
        if current_obj > best_obj:
            best_obj = current_obj
            best_v = binary_v
        
        if np.abs(last_v_obj - current_obj) / np.abs(last_v_obj) < 1e-3:
            break
        last_v = v

    pred_x = best_v.reshape((n2, n1)).T
    pred_x = hungarian(pred_x)
    task_data.from_data(sol=pred_x.reshape((n1n2,)), ref=False)

   
    
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