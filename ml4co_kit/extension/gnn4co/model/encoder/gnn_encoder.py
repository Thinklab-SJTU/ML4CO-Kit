r"""
GNN Encoder.
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


import copy
from torch import nn, Tensor
from typing import Sequence, Union
from ml4co_kit.task.base import TASK_TYPE
from ..embedder import get_embedder_by_task
from ..out_layer import get_out_layer_by_task
from .gnn_layer import GNNSparseBlock, GNNDenseBlock


class GNNEncoder(nn.Module):
    def __init__(
        self,
        task_type: TASK_TYPE,
        sparse: bool,
        block_layers: Sequence[int],
        hidden_dim: int = 256, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False
    ):
        super(GNNEncoder, self).__init__()
        
        # embedder and out_layer
        self.task_type = task_type
        self.embedder = get_embedder_by_task(task_type)(hidden_dim, sparse)
        self.out_layer = get_out_layer_by_task(task_type)(hidden_dim, 2, sparse)

        # asym
        self.asym = True if task_type == TASK_TYPE.ATSP else False

        # gnn blocks
        if sparse:
            # gnn sparse blocks
            self.blocks = nn.ModuleList([
                GNNSparseBlock(
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    aggregation=aggregation,
                    norm=norm,
                    learn_norm=learn_norm,
                    track_norm=track_norm,
                    asym=self.asym
                ) for num_layers in block_layers
            ])
        else:
            # gnn dense blocks
            self.blocks = nn.ModuleList([
                GNNDenseBlock(
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    aggregation=aggregation,
                    norm=norm,
                    learn_norm=learn_norm,
                    track_norm=track_norm,
                    asym=self.asym
                ) for num_layers in block_layers
            ])
            
    def forward(
        self, x: Tensor, e: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        if self.asym:
            return self.asym_forward(x=x, e=e, edge_index=edge_index)
        else:
            return self.sym_forward(x=x, e=e, edge_index=edge_index)
        
    def asym_forward(
        self, x: Tensor, e: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        # nodes number
        nodes_num = None if x is None else x.shape[0]
        
        # edges feature
        edges_feature = copy.deepcopy(e)
        
        # embedder
        e = self.embedder(x, e)
        
        # gnn blocks
        for gnn_block in self.blocks:
            gnn_block: Union[GNNDenseBlock, GNNSparseBlock]
            e = gnn_block.asym_forward(
                e=e, edges_feature=edges_feature, 
                edge_index=edge_index, nodes_num=nodes_num
            )

        # out layer
        x, e = self.out_layer(x, e)
        
        # return
        return x, e
    
    def sym_forward(
        self, x: Tensor, e: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        # embedder
        x, e = self.embedder(x, e)

        # gnn blocks
        for gnn_block in self.blocks:
            gnn_block: Union[GNNDenseBlock, GNNSparseBlock]
            x, e = gnn_block.forward(x=x, e=e, edge_index=edge_index)

        # out layer
        x, e = self.out_layer(x, e)

        # return
        return x, e