r"""
GNN4CO Denser.
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
from typing import List
from torch import Tensor
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.task.routing.atsp import ATSPTask
from ml4co_kit.utils.type_utils import to_tensor
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class DenseData(object):
    def __init__(
        self, 
        node_feature: Tensor, 
        graph: Tensor, 
        ground_truth: Tensor
    ):
        """
        Args:
            node_feature: Node features (V, D_v)
                - The feature of each node
            graph: Graph (V, V)
                - The feature of each edge
            ground_truth: Ground truth (V) or (E)
                - The ground truth of the graph
        """
        super(DenseData, self).__init__()
        self.node_feature = node_feature
        self.graph = graph
        self.ground_truth = ground_truth


class DenseDataBatch(object):
    def __init__(self):
        # Initialize lists
        self.node_feature_list = list()
        self.graph_list = list()
        self.ground_truth_list = list()

        # Initialize variables
        self.node_feature = None
        self.graph = None
        self.ground_truth = None

    def from_data_list(self, data_list: List[DenseData]):
        # Add data to lists
        for data in data_list:
            self.node_feature_list.append(data.node_feature)
            self.graph_list.append(data.graph)
            self.ground_truth_list.append(data.ground_truth)

        # Stack data
        self.node_feature = torch.stack(self.node_feature_list, 0)
        self.graph = torch.stack(self.graph_list, 0)
        self.ground_truth = torch.stack(self.ground_truth_list, 0)
        
    def to_cuda(self):
        self.node_feature = self.node_feature.cuda()
        self.graph = self.graph.cuda()
        self.ground_truth = self.ground_truth.cuda()


class GNN4CODenser(object):
    def __init__(self, device: str) -> None:
        self.device = device
        self.dense_process_func_dict = {
            TASK_TYPE.ATSP: self.atsp_dense_process,
            TASK_TYPE.TSP: self.tsp_dense_process
        }

    def batch_data_process(
        self, batch_task_data: List[TaskBase], sampling_num: int = 1
    ) -> DenseDataBatch:
        # Preparation
        self.dense_data_list = list()
        task_type = batch_task_data[0].task_type
        dense_process_func = self.dense_process_func_dict[task_type]

        # Dense process
        for task_data in batch_task_data:
            dense_data = dense_process_func(task_data)
            for _ in range(sampling_num):
                self.dense_data_list.append(dense_data)

        # Merge
        batch_dense_data = DenseDataBatch()
        batch_dense_data.from_data_list(self.dense_data_list)
        if self.device == "cuda":
            batch_dense_data.to_cuda()
        return batch_dense_data

    def tsp_dense_process(self, task_data: TSPTask) -> DenseData:
        # Nodes feature (V, 2)
        nodes_num: int = task_data.nodes_num
        points = task_data.points
        nodes_feature = to_tensor(points)

        # Graph (V, V)
        graph = to_tensor(task_data._get_dists()).float()

        # Ground truth (V, V)
        if task_data.ref_sol is not None:
            ref_tour = task_data.ref_sol
            ground_truth = torch.zeros(size=(nodes_num, nodes_num))
            for idx in range(len(ref_tour) - 1):
                ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
            ground_truth = ground_truth + ground_truth.T
        else:
            ground_truth = torch.zeros(size=(nodes_num, nodes_num))

        # Return DenseData
        dense_data = DenseData(
            node_feature=nodes_feature.float(),
            graph=graph.float(),
            ground_truth=ground_truth.long()
        )
        return dense_data

    def atsp_dense_process(self, task_data: ATSPTask) -> DenseData:
        # Nodes feature (V, 2)
        nodes_num: int = task_data.nodes_num
        nodes_feature = torch.zeros(size=(nodes_num, 2))

        # Graph (V, V)
        graph = to_tensor(task_data.dists).float()

        # Ground truth (V, V)
        if task_data.ref_sol is not None:
            ref_tour = task_data.ref_sol
            ground_truth = torch.zeros(size=(nodes_num, nodes_num))
            for idx in range(len(ref_tour) - 1):
                ground_truth[ref_tour[idx]][ref_tour[idx+1]] = 1
        else:
            ground_truth = torch.zeros(size=(nodes_num, nodes_num))

        # Return DenseData
        dense_data = DenseData(
            node_feature=nodes_feature.float(),
            graph=graph.float(),
            ground_truth=ground_truth.long()
        )
        return dense_data