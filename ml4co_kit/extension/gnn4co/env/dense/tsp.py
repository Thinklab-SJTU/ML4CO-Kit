r"""
Dense Process for TSP.
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
from scipy.spatial.distance import cdist
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.utils.type_utils import to_tensor


def tsp_dense_process(task_data: TSPTask):
    # Extract Data
    points = task_data.points
    ref_tour = task_data.ref_sol
    
    # nodes_num
    nodes_num = points.shape[0]
    
    # x and graph
    x = to_tensor(points)
    graph = to_tensor(cdist(points, points)).float()
    
    # ground truth
    if ref_tour is not None:
        ground_truth = torch.zeros(size=(nodes_num, nodes_num))
        for idx in range(len(ref_tour) - 1):
            ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        ground_truth = (ground_truth + ground_truth.T).long()
    else:
        ground_truth = None

    return (
        x, # (V, 2): nodes feature, random init
        graph.float(), # (V, V): edges feature, distance matrix
        ground_truth, # (V,): Ground truth
        nodes_num, # Number of nodes
    )