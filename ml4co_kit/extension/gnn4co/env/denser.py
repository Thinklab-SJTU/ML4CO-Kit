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
from typing import List, Any
from ml4co_kit.task.graph.mcl import MClTask
from ml4co_kit.task.graph.mis import MISTask
from ml4co_kit.task.graph.mvc import MVCTask
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.task.graph.mcut import MCutTask
from ml4co_kit.task.routing.atsp import ATSPTask
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from .dense import (
    atsp_dense_process, cvrp_dense_process, tsp_dense_process
)


class GNN4CODenser(object):
    def __init__(self, device: str) -> None:
        self.device = device
    
    #################################
    #        Raw Data Process       #
    #################################
    
    def initial_lists(self):
        self.nodes_feature_list = list()
        self.x_list = list()
        self.graph_list = list()
        self.ground_truth_list = list()
        self.nodes_num_list = list()
        
    def update_lists(self, dense_data: Any):
        self.x_list.append(dense_data[0])
        self.graph_list.append(dense_data[1])
        self.ground_truth_list.append(dense_data[2])
        self.nodes_num_list.append(dense_data[3])
    
    def edge_merge_process(self, task: str, with_gt: bool) -> Any:
        # nodes feature
        if self.x_list[0] is not None:
            x = torch.stack(self.x_list, 0).to(self.device)
        else:
            x = None
            
        # graph
        graph = torch.stack(self.graph_list, 0).to(self.device)

        # ground truth
        if with_gt:
            ground_truth = torch.stack(
                self.ground_truth_list, 0
            ).to(self.device) # (B, V, V) or (B, V)
        else:
            ground_truth = None
        return (task, x, graph, ground_truth, self.nodes_num_list)  
        
    def node_merge_process(self, task: str, with_gt: bool) -> Any:
        raise NotImplementedError()
    
    def data_process(
        self, batch_task_data: List[TaskBase], sampling_num: int = 1
    ) -> Any:
        task_data = batch_task_data[0]
        if task_data.task_type == TASK_TYPE.ATSP:
            return self.atsp_batch_data_process(batch_task_data, sampling_num)
        elif task_data.task_type == TASK_TYPE.CVRP:
            return self.cvrp_batch_data_process(batch_task_data, sampling_num)
        elif task_data.task_type == TASK_TYPE.TSP:
            return self.tsp_batch_data_process(batch_task_data, sampling_num)
        else:
            raise NotImplementedError()
    
    def atsp_batch_data_process(
        self, batch_task_data: List[ATSPTask], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # dense process
        for idx in range(len(batch_task_data)):
            dense_data = atsp_dense_process(task_data=batch_task_data[idx])
            for _ in range(sampling_num):
                self.update_lists(dense_data)
            
        # merge
        return self.edge_merge_process(task="ATSP", with_gt=with_gt)
           
    def cvrp_batch_data_process(
        self, batch_task_data: List[CVRPTask], sampling_num: int = 1    
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # dense process
        for idx in range(len(batch_task_data)):
            dense_data = cvrp_dense_process(task_data=batch_task_data[idx])
            for _ in range(sampling_num):
                self.update_lists(dense_data)
        
        # merge
        return self.edge_merge_process(task="CVRP", with_gt=with_gt)

    def mcl_batch_data_process(
        self, batch_task_data: List[MClTask], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError("MCl is not supported currently.")
    
    def mcut_batch_data_process(
        self, batch_task_data: List[MCutTask], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError("MCut is not supported currently.")

    def mis_batch_data_process(
        self, batch_task_data: List[MISTask], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError("MIS is not supported currently.")

    def mvc_batch_data_process(
        self, batch_task_data: List[MVCTask], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError("MVC is not supported currently.")
    
    def tsp_batch_data_process(
        self, batch_task_data: List[TSPTask], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # dense process
        for idx in range(len(batch_task_data)):
            dense_data = tsp_dense_process(task_data=batch_task_data[idx])
            for _ in range(sampling_num):
                self.update_lists(dense_data)
        
        # merge
        return self.edge_merge_process(task="TSP", with_gt=with_gt)