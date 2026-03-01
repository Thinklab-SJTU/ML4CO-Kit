r"""
Pygmtools Solver for Graph Matching.
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


import torch
from typing import List
from torch import Tensor
from ml4co_kit.task.qap.gm import GMTask
from ml4co_kit.utils import to_tensor, to_numpy
from ml4co_kit.extension.pygmtools import PyGMToolsQAPSolver, pygm_hungarian


def gm_pygm(
    task_data: GMTask, pygm_qap_solver: PyGMToolsQAPSolver
):
    return gm_pygm_batch(
        batch_task_data=[task_data], pygm_qap_solver=pygm_qap_solver
    )


def gm_pygm_batch(
    batch_task_data: List[GMTask], pygm_qap_solver: PyGMToolsQAPSolver
):
    # Merge affinity matrices
    Ks = []
    n1s = []
    n2s = []
    max_n1n2 = 0
    batch_size = len(batch_task_data)
    for task_data in batch_task_data:
        Ks.append(task_data.K)
        n1s.append(task_data.n1)
        n2s.append(task_data.n2)

    # Convert to tensor with proper index mapping
    n1_tensor = torch.tensor(n1s)
    n2_tensor = torch.tensor(n2s)
    max_n1 = max(n1s)
    max_n2 = max(n2s)
    max_n1n2 = max_n1 * max_n2
    K_tensor = torch.zeros(batch_size, max_n1n2, max_n1n2)
    
    for batch_idx, K in enumerate(Ks):
        n1 = n1s[batch_idx]
        n2 = n2s[batch_idx]
        
        # K[i*n2+j, p*n2+q] should map to K_tensor[i*max_n2+j, p*max_n2+q]
        # Use reshape to avoid loops
        # Step 1: Reshape K from (n1*n2, n1*n2) to (n1, n2, n1, n2)
        K: Tensor
        K_4d = K.reshape(n1, n2, n1, n2)
        
        # Step 2: Place into padded 4D tensor (max_n1, max_n2, max_n1, max_n2)
        K_padded_4d = torch.zeros(max_n1, max_n2, max_n1, max_n2)
        K_padded_4d[:n1, :n2, :n1, :n2] = to_tensor(K_4d)
        
        # Step 3: Reshape back to 2D (max_n1*max_n2, max_n1*max_n2)
        K_tensor[batch_idx] = K_padded_4d.reshape(max_n1n2, max_n1n2)
    
    # Solve
    Xs = pygm_qap_solver.solve(
        K=K_tensor, n1=n1_tensor, n2=n2_tensor, n1max=None, n2max=None
    )
    Xs = to_numpy(pygm_hungarian(Xs))
    for task_data, X, n1, n2 in zip(batch_task_data, Xs, n1s, n2s):
        task_data: GMTask
        task_data.from_data(sol=X[:n1, :n2], ref=False)
    return batch_task_data