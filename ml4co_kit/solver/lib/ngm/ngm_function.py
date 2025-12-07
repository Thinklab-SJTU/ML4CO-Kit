r"""
NGM Algorithm for GM
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
import numpy as np
from torch import nn
from typing import Tuple
from ml4co_kit.task.graphset.gm import GMTask
from ml4co_kit.solver.lib.utils_gm import hungarian
from ml4co_kit.solver.lib.ngm.ngm_torch.func_torch import ngm_torch


def ngm(
    batch_task_data: list[GMTask],
    x0: np.ndarray = None,
    gnn_channels: Tuple[int, ...] = (16, 16, 16),
    sk_emb: int = 1,
    sk_max_iter: int = 50,
    sk_tau: float = 0.05,
    network: nn.Module = None,
    pretrain: str = "voc",
    device: str = "cpu",
):
    # Preparation (data)
    Ks, n1s, n2s = [], [], [] 
    n1max, n2max = 0, 0
    for task_data in batch_task_data:
        if task_data.aff_mat is None:
            task_data.aff_mat = task_data.build_aff_mat()
        K = task_data.aff_mat
        n1 = task_data.graphs[0].nodes_num
        n2 = task_data.graphs[1].nodes_num
        Ks.append(K)
        n1s.append(int(n1))
        n2s.append(int(n2))
    n1max = max(n1s)
    n2max = max(n2s)
    Nmax = n1max * n2max
    
    K_batch = []
    for K_np, n1, n2 in zip(Ks, n1s, n2s):
        Ni = n1 * n2
        K_pad = np.pad(K_np, ((0, Nmax - Ni), (0, Nmax - Ni)), mode='constant')
        K_batch.append(torch.from_numpy(K_pad))
    K_batch = torch.stack(K_batch, dim=0).to(device) 
    x0_batch = None if x0 is None else torch.from_numpy(x0).to(device=device)

    n1_tensor = torch.tensor(n1s, dtype=torch.int64).to(device) # (B,)
    n2_tensor = torch.tensor(n2s, dtype=torch.int64).to(device) # (B,)
            
    sol_soft, _ = ngm_torch(
        K=K_batch,
        n1=n1_tensor,
        n2=n2_tensor,
        n1max=n1max,
        n2max=n2max,
        x0=x0_batch,
        gnn_channels=gnn_channels,
        sk_emb=sk_emb,
        sk_max_iter=sk_max_iter,
        sk_tau=sk_tau,
        network=network,
        pretrain=pretrain
    )
    
    X_soft = sol_soft.detach().cpu().numpy()
    X_soft = X_soft.reshape(-1, n1max, n2max)
    
    batch = X_soft.shape[0]
    for i in range(batch):
        X = hungarian(X_soft[i][:n1s[i], :n2s[i]])
        batch_task_data[i].from_data(sol=X, ref=False)
    