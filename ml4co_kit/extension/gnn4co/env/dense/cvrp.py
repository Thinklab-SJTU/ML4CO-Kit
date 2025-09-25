r"""
Dense Process for CVRP.
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
import numpy as np
from torch import Tensor
from typing import Sequence
from scipy.spatial.distance import cdist
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.utils.type_utils import to_tensor


def cvrp_dense_process(task_data: CVRPTask) -> Sequence[Tensor]:
    # Extract Data
    depot = task_data.depots
    points = task_data.points
    demand = task_data.demands
    ref_tour = task_data.ref_sol
    
    # nodes num
    nodes_num = len(demand) + 1
    
    # update points
    depots = depot.reshape(1, 2)
    points = np.concatenate([depots, points], 0)

    # graph
    graph = to_tensor(cdist(points, points)).float()
    e = torch.zeros_like(graph)
    
    # update demand
    demand_new = np.zeros(shape=(nodes_num,))
    demand_new[1:] = demand
    demand_new = demand_new.reshape(nodes_num, 1)
    
    # ground truth (partition matrix)
    if ref_tour is not None:   
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = ground_truth + ground_truth.T
        ground_truth = torch.clip(ground_truth, 0, 1).long()
    else:
        ground_truth = None

    # node feature
    depot_flag = np.zeros(shape=(nodes_num,))
    depot_flag[0] = 1
    depot_flag = depot_flag.reshape(nodes_num, 1)
    x = np.concatenate([points, demand_new, depot_flag], 1)
    nodes_feature = to_tensor(x)

    return (
        nodes_feature.float(), # (V, 4): nodes feature
        graph.float(), # (V, V): edges feature, distance matrix
        ground_truth, # (V,): Ground truth
        nodes_num, # Number of nodes
    )