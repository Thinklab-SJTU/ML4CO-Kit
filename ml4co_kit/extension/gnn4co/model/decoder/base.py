r"""
GNN4CO Decoder Base.
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


import numpy as np
import scipy.sparse
from torch import Tensor
from typing import Union, List
from ml4co_kit.utils.type_utils import to_numpy
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.task.graph.base import GraphTaskBase
from ...env import SparseDataBatch, DenseDataBatch


class GNN4CODecoder(object):
    def __init__(self, sparse_factor: int) -> None:
        self.sparse_factor = sparse_factor
        
    def sparse_decode(
        self, 
        heatmap: Tensor, 
        task_type: TASK_TYPE, 
        batch_task_data: List[TaskBase], 
        batch_processed_data: SparseDataBatch,
        return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        # Data
        heatmap = to_numpy(heatmap)
        edge_index = to_numpy(batch_processed_data.edge_index)

        # TSP heatmap to dense heatmap
        if task_type == TASK_TYPE.TSP:
            accumulated_nodes_num = 0
            edge_begin_idx = 0
            for task_data in batch_task_data:
                task_data: TSPTask
                nodes_num = task_data.nodes_num
                edge_end_idx = edge_begin_idx + nodes_num * self.sparse_factor
                _heatmap = heatmap[edge_begin_idx:edge_end_idx]
                _edge_index = edge_index[:, edge_begin_idx:edge_end_idx]
                _edge_index = _edge_index - accumulated_nodes_num
                _dense_heatmap: np.ndarray = scipy.sparse.coo_matrix(
                    arg1=(_heatmap, (_edge_index[0], _edge_index[1])), 
                    shape=(nodes_num, nodes_num)
                ).toarray()
                _dense_heatmap = (_dense_heatmap + _dense_heatmap.T) / 2
                task_data.cache["heatmap"] = _dense_heatmap
                edge_begin_idx = edge_end_idx
                accumulated_nodes_num += nodes_num

        # Graph heatmap
        elif task_type in [TASK_TYPE.MIS, TASK_TYPE.MCUT, TASK_TYPE.MCL, TASK_TYPE.MVC]:
            node_begin_idx = 0
            for task_data in batch_task_data:
                task_data: GraphTaskBase
                nodes_num = task_data.nodes_num
                node_end_idx = node_begin_idx + nodes_num
                _heatmap = heatmap[node_begin_idx:node_end_idx]
                task_data.cache["heatmap"] = _heatmap
                node_begin_idx = node_end_idx
        else:
            raise NotImplementedError()

        # Decode
        if return_cost:
            costs = list()
        for task_data in batch_task_data:
            self._decode(task_data)
            if return_cost:
                costs.append(task_data.evaluate(task_data.sol))
        if return_cost:
            return np.mean(costs)

    def dense_decode(
        self, 
        heatmap: Tensor, 
        task_type: TASK_TYPE, 
        batch_task_data: List[TaskBase], 
        batch_processed_data: DenseDataBatch,
        return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        # If return cost, initialize costs list
        if return_cost:
            costs = list()

        # Decode
        for _heatmap, task_data in zip(heatmap, batch_task_data):
            task_data.cache["heatmap"] = to_numpy(_heatmap)
            self._decode(task_data)
            if return_cost:
                costs.append(task_data.evaluate(task_data.sol))
        
        # If return cost, return the mean of costs
        if return_cost:
            return np.mean(costs)

    def _decode(self, task_data: TaskBase):
        raise NotImplementedError("Subclasses should implement this method.")