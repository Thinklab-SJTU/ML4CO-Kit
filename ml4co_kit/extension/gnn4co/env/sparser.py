r"""
GNN4CO Sparser.
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
from .sparse import (
    atsp_sparse_process, mcl_sparse_process, mcut_sparse_process, 
    mis_sparse_process, mvc_sparse_process, tsp_sparse_process
)


class GNN4COSparser(object):
    def __init__(self, sparse_factor: int, device: str) -> None:
        self.sparse_factor = sparse_factor
        self.device = device
    
    def initial_lists(self):
        self.x_list = list()
        self.e_list = list()
        self.edge_index_list = list()
        self.graph_list = list()
        self.ground_truth_list = list()
        self.nodes_num_list = list()
        self.edges_num_list = list()
        
    def update_lists(self, sparse_data: Any):
        self.x_list.append(sparse_data[0])
        self.e_list.append(sparse_data[1])
        self.edge_index_list.append(sparse_data[2])
        self.graph_list.append(sparse_data[3])
        self.ground_truth_list.append(sparse_data[4])
        self.nodes_num_list.append(sparse_data[5])
        self.edges_num_list.append(sparse_data[6])
    
    def merge_process(self, task: str, with_gt: bool) -> Any:
        # nodes feature
        if self.x_list[0] is not None:
            x = torch.cat(self.x_list, 0).to(self.device) # (V, C) or (V,)
        else:
            x = None
            
        # edges feature
        if self.e_list[0] is not None:
            e = torch.cat(self.e_list, 0).to(self.device) # (V, C) or (E,)
        else:
            e = None

        # edge index
        add_index = 0
        edge_index_list = list()
        for idx, edge_index in enumerate(self.edge_index_list):
            edge_index_list.append(edge_index + add_index)
            add_index += self.nodes_num_list[idx]
        edge_index = torch.cat(edge_index_list, 1).to(self.device) # (2, E)

        # ground truth
        if with_gt:
            ground_truth = torch.cat(self.ground_truth_list, 0).to(self.device) # (E,) or (V,)
        else:
            ground_truth = None
            
        return (
            task, x, e, edge_index, self.graph_list, 
            ground_truth, self.nodes_num_list, self.edges_num_list
        )
    
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
        elif task_data.task_type == TASK_TYPE.MCL:
            return self.mcl_batch_data_process(batch_task_data, sampling_num)
        elif task_data.task_type == TASK_TYPE.MCUT:
            return self.mcut_batch_data_process(batch_task_data, sampling_num)
        elif task_data.task_type == TASK_TYPE.MIS:
            return self.mis_batch_data_process(batch_task_data, sampling_num)
        elif task_data.task_type == TASK_TYPE.MVC:
            return self.mvc_batch_data_process(batch_task_data, sampling_num)
        else:
            raise NotImplementedError("Task type is not supported currently.")
    
    def atsp_batch_data_process(
        self, batch_task_data: List[ATSPTask], sampling_num: int = 1
    ) -> Any:        
        # initialize lists
        self.initial_lists()
        
        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # sparse process
        for idx in range(len(batch_task_data)):
            sparse_data = atsp_sparse_process(
                task_data=batch_task_data[idx], 
                sparse_factor=self.sparse_factor
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
            
        # merge
        return self.merge_process(task="ATSP", with_gt=with_gt)
           
    def cvrp_batch_data_process(
        self, batch_task_data: List[CVRPTask], sampling_num: int = 1
    ) -> Any:
        raise NotImplementedError(
            "CVRP is not supported currently."
        )
    
    def mcl_batch_data_process(
        self, batch_task_data: List[MClTask], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # sparse process
        for graph in batch_task_data:
            sparse_data = mcl_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process(task="MCl", with_gt=with_gt)
    
    def mcut_batch_data_process(
        self, batch_task_data: List[MCutTask], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # sparse process
        for graph in batch_task_data:
            sparse_data = mcut_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process(task="MCut", with_gt=with_gt)

    def mis_batch_data_process(
        self, batch_task_data: List[MISTask], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # sparse process
        for graph in batch_task_data:
            sparse_data = mis_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process(task="MIS", with_gt=with_gt)

    def mvc_batch_data_process(
        self, batch_task_data: List[MVCTask], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # sparse process
        for graph in batch_task_data:
            sparse_data = mvc_sparse_process(graph)
            for _ in range(sampling_num):
                self.update_lists(sparse_data)

        # merge
        return self.merge_process(task="MVC", with_gt=with_gt)
    
    def tsp_batch_data_process(
        self, batch_task_data: List[TSPTask], sampling_num: int = 1
    ) -> Any:
        # initialize lists
        self.initial_lists()

        # with_gt
        with_gt = True if batch_task_data[0].ref_sol is not None else False
        
        # sparse process
        for idx in range(len(batch_task_data)):
            sparse_data = tsp_sparse_process(
                task_data=batch_task_data[idx], 
                sparse_factor=self.sparse_factor
            )
            for _ in range(sampling_num):
                self.update_lists(sparse_data)
        
        # merge
        return self.merge_process(task="TSP", with_gt=with_gt)