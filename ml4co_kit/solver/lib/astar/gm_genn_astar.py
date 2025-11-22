r"""
GENN AStar Solver.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import torch.nn as nn
import numpy as np
from ml4co_kit.task.graphset.base import hungarian
from ml4co_kit.task.graphset.gm import GMTask
from ml4co_kit.utils import download
from .modules import default_parameter, GraphPair, GENN, check_layer_parameter, _load_model, astar_pretrain_path, genn_astar_kernel



def gm_genn_astar(
    task_data: list[GMTask],
    channel: int = None, 
    filters_1: int = 64, 
    filters_2: int = 32, 
    filters_3: int = 16,
    tensor_neurons: int = 16, 
    beam_width: int = 0, 
    trust_fact: float = 1, 
    no_pred_size: int = 0, 
    network: nn.Module = None, 
    pretrain: str = "AIDS700nef",
    device: str = "cpu"
    ):
    """
    Pytorch implementation of GENN-ASTAR
    """
    device = torch.device(device)
    
    feat1, feat2 = [], []
    A1, A2=  [], []
    n1, n2 = [], []
    max_n1 = max(task.graphs[0].nodes_num for task in task_data)
    max_n2 = max(task.graphs[1].nodes_num for task in task_data)
    precision = torch.float32 if task_data[0].precision == np.float32 else torch.float64
    for task in task_data:
        g1, g2 = task.graphs[0], task.graphs[1]
        
        A1_real = g1.to_adj_matrix()
        A2_real = g2.to_adj_matrix()
        
        feat1_real = g1.nodes_feature
        feat2_real = g2.nodes_feature
        
        feat1_pad = np.pad(feat1_real, ((0, max_n1 - feat1_real.shape[0]), (0, 0)), mode='constant')
        feat2_pad = np.pad(feat2_real, ((0, max_n2 - feat2_real.shape[0]), (0, 0)), mode='constant')
        
        A1_pad = np.pad(A1_real, ((0, max_n1 - A1_real.shape[0]), (0, max_n1 - A1_real.shape[1])), mode='constant')
        A2_pad = np.pad(A2_real, ((0, max_n2 - A2_real.shape[0]), (0, max_n2 - A2_real.shape[1])), mode='constant')
        
        n1.append(g1.nodes_num)
        n2.append(g2.nodes_num)
        feat1.append(feat1_pad)
        feat2.append(feat2_pad)
        A1.append(A1_pad)
        A2.append(A2_pad)
        
    feat1_batch = torch.tensor(np.stack(feat1), dtype=precision, device=device)
    feat2_batch = torch.tensor(np.stack(feat2), dtype=precision, device=device)
    A1_batch = torch.tensor(np.stack(A1), dtype=precision, device=device)
    A2_batch = torch.tensor(np.stack(A2), dtype=precision, device=device)
    n1_torch = torch.tensor(n1, dtype=int, device=device)
    n2_torch = torch.tensor(n2, dtype=int, device=device)
    
    batch_X, _ = genn_astar_kernel(
        feat1=feat1_batch, 
        feat2=feat2_batch, 
        A1=A1_batch, 
        A2=A2_batch, 
        n1=n1_torch, 
        n2=n2_torch, 
        channel=channel, 
        filters_1=filters_1, 
        filters_2=filters_2, 
        filters_3=filters_3,
        tensor_neurons=tensor_neurons, 
        beam_width=beam_width, 
        trust_fact=trust_fact, 
        no_pred_size=no_pred_size, 
        network=network, 
        pretrain=pretrain, 
        use_net=True
        )
    
    batch_size = len(task_data)
    for i in range(batch_size):
        X = batch_X[i][:n1[i], :n2[i]].detach().cpu().numpy()
        X = hungarian(X)
        task_data[i].from_data(sol=X, ref=False)
  