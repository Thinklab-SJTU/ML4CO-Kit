r"""
Pygmtools Solver for Graph Edit Distance.
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
from ml4co_kit.task.qap.ged import GEDTask
from ml4co_kit.utils import to_tensor, to_numpy
from ml4co_kit.extension.pygmtools import PyGMToolsQAPSolver, pygm_hungarian


def ged_pygm(
    task_data: GEDTask, pygm_qap_solver: PyGMToolsQAPSolver, temperature: float = 0.3
):
    return ged_pygm_batch(
        batch_task_data=[task_data], pygm_qap_solver=pygm_qap_solver, temperature=temperature
    )


def ged_pygm_batch(
    batch_task_data: List[GEDTask], pygm_qap_solver: PyGMToolsQAPSolver, temperature: float = 0.3
):
    # Merge affinity matrices
    Ks = []
    n1s = []
    n2s = []
    max_n1n2 = 0
    batch_size = len(batch_task_data)
    for task_data in batch_task_data:
        import numpy as np
        Ks.append(task_data.K)
        n1s.append(task_data.n1)
        n2s.append(task_data.n2)

    # Convert to tensor with proper index mapping
    n1_tensor = torch.tensor(n1s)
    n2_tensor = torch.tensor(n2s)
    max_n1 = max(n1s)
    max_n2 = max(n2s)
    max_n2n1 = max_n2 * max_n1
    K_tensor = torch.zeros(batch_size, max_n2n1, max_n2n1)
    
    for batch_idx, K in enumerate(Ks):
        n1 = n1s[batch_idx]
        n2 = n2s[batch_idx]
        
        # K[j*n1+i, q*n1+p] should map to K_tensor[j*max_n2+i, q*max_n2+p]
        # Use reshape to avoid loops
        # Step 1: Reshape K from (n2*n1, n2*n1) to (n2, n1, n2, n1)
        K: Tensor
        K_4d = K.reshape(n2, n1, n2, n1)
        
        # Step 2: Place into padded 4D tensor (max_n2, max_n1, max_n2, max_n1)
        K_padded_4d = torch.zeros(max_n2, max_n1, max_n2, max_n1)
        K_padded_4d[:n2, :n1, :n2, :n1] = to_tensor(K_4d)
        
        # Step 3: Reshape back to 2D (max_n2*max_n1, max_n2*max_n1)
        K_tensor[batch_idx] = K_padded_4d.reshape(max_n2n1, max_n2n1)
    
    # Convert cost to affinity with temperature scaling
    if pygm_qap_solver.solver_name in ["ipfp", "rrwm", "sm"]:
        K_tensor = torch.exp(-K_tensor / temperature)  
    elif pygm_qap_solver.solver_name == "astar":
        K_tensor = -K_tensor
        pygm_qap_solver.is_dummy = True  # Enable dummy node handling for A* solver
    else:
        raise ValueError(f"Unsupported pygm_qap_solver: {pygm_qap_solver.solver_name}")
    
    # Solve
    Xs = pygm_qap_solver.solve(
        K=K_tensor, n1=n1_tensor, n2=n2_tensor, n1max=None, n2max=None
    )
    
    if pygm_qap_solver.solver_name in ["ipfp", "rrwm", "sm"]:
        Xs = pygm_hungarian(Xs)
        
    Xs = to_numpy(Xs)

    for task_data, X, n1, n2 in zip(batch_task_data, Xs, n1s, n2s):
        # mask dummy node pairs
        n_g1 = task_data.g1.nodes_num
        n_g2 = task_data.g2.nodes_num
        X[n_g1:, n_g2:] = 0
        
        task_data: GEDTask
        task_data.from_data(sol=X[:n1, :n2], ref=False)
        
    return batch_task_data