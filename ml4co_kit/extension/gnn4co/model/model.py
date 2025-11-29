r"""
GNN4CO Model.
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


import os
import torch
from typing import Any
from torch import Tensor, nn
from typing import Union, Tuple
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.learning.model import BaseModel
from ml4co_kit.utils.file_utils import pull_file_from_huggingface
from .decoder.base import GNN4CODecoder
from .encoder.gnn_encoder import GNNEncoder
from ..env.env import GNN4COEnv, DenseDataBatch, SparseDataBatch


SUPPORTS = [
    "gnn4co_mcl_rb-large_sparse.pt",
    "gnn4co_mcl_rb-small_sparse.pt",
    "gnn4co_mcut_ba-large_sparse.pt",
    "gnn4co_mcut_ba-small_sparse.pt",
    "gnn4co_mis_er-700-800_sparse.pt",
    "gnn4co_mis_rb-large_sparse.pt",
    "gnn4co_mis_rb-small_sparse.pt",
    "gnn4co_mis_satlib_sparse.pt",
    "gnn4co_mvc_rb-large_sparse.pt",
    "gnn4co_mvc_rb-small_sparse.pt",
    "gnn4co_tsp1k_sparse.pt",
    "gnn4co_tsp10k_sparse.pt",
    "gnn4co_tsp50_dense.pt",
    "gnn4co_tsp100_dense.pt",
    "gnn4co_tsp500_sparse.pt",
    "gnn4co_atsp50_dense.pt",
    "gnn4co_atsp100_dense.pt",
    "gnn4co_atsp200_dense.pt",
    "gnn4co_atsp500_dense.pt"
]


class GNN4COModel(BaseModel):
    def __init__(
        self,
        env: GNN4COEnv,
        encoder: GNNEncoder,
        decoder: GNN4CODecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        weight_path: str = None
    ):
        super(GNN4COModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: GNN4COEnv
        self.model: GNNEncoder
        self.decoder: GNN4CODecoder = decoder

        # load pretrained weights if needed
        if weight_path is not None:
            if not os.path.exists(weight_path):
                self.download_weight(weight_path)
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # Set mode
        self.env.mode = phase
        
        # Get real data
        if phase == "train":
            # Get real train batch data
            batch_size = len(batch)
            batch_task_data, batch_processed_data = self.env.generate_train_data(batch_size)
            task_type = batch_task_data[0].task_type

            # Deal with different task
            if task_type in [TASK_TYPE.TSP, TASK_TYPE.ATSP]:
                if self.env.sparse:
                    loss = self.train_edge_sparse(batch_processed_data)
                else:
                    loss = self.train_edge_dense(batch_processed_data)
            elif task_type in [TASK_TYPE.MIS, TASK_TYPE.MCUT, TASK_TYPE.MCL, TASK_TYPE.MVC]:
                loss = self.train_node_sparse(batch_processed_data)
            else:
                raise NotImplementedError()
            
        elif phase == "val":
            # Get val data
            batch_task_data, batch_processed_data = self.env.generate_val_data(batch_idx)
            task_type = batch_task_data[0].task_type

            # Deal with different task
            if task_type in [TASK_TYPE.TSP, TASK_TYPE.ATSP]:
                if self.env.sparse:
                    loss, heatmap = self.inference_edge_sparse(batch_processed_data)
                else:
                    loss, heatmap = self.inference_edge_dense(batch_processed_data)
            elif task_type in [TASK_TYPE.MIS, TASK_TYPE.MCUT, TASK_TYPE.MCL, TASK_TYPE.MVC]:
                loss, _ = self.inference_node_sparse(batch_processed_data)
            else:
                raise NotImplementedError()

            # Decoding
            if self.env.sparse:
                costs_avg = self.decoder.sparse_decode(
                    heatmap=heatmap, task_type=task_type, batch_task_data=batch_task_data, 
                    batch_processed_data=batch_processed_data, return_cost=True
                )
            else:
                costs_avg = self.decoder.dense_decode(
                    heatmap=heatmap, task_type=task_type, batch_task_data=batch_task_data, 
                    batch_processed_data=batch_processed_data, return_cost=True
                )
        
        else:
            raise NotImplementedError()
        
        # Log metrics
        metrics = {f"{phase}/loss": loss}
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg})
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Return
        return loss if phase == "train" else metrics   

    def train_edge_sparse(
        self, batch_processed_data: SparseDataBatch
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            nodes_feature = batch_processed_data.node_feature, 
            edges_feature = batch_processed_data.edge_feature, 
            edge_index = batch_processed_data.edge_index
        )
        del x_pred
        loss = nn.CrossEntropyLoss()(e_pred, batch_processed_data.ground_truth)
        return loss
    
    def train_edge_dense(
        self, batch_processed_data: DenseDataBatch
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            x=batch_processed_data.node_feature, 
            e=batch_processed_data.graph,
            edge_index=None
        )
        del x_pred
        loss = nn.CrossEntropyLoss()(e_pred, batch_processed_data.ground_truth)
        return loss

    def train_node_sparse(
        self, batch_processed_data: SparseDataBatch
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            x=batch_processed_data.node_feature, 
            e=batch_processed_data.edge_feature, 
            edge_index=batch_processed_data.edge_index
        )
        del e_pred
        loss = nn.CrossEntropyLoss()(x_pred, batch_processed_data.ground_truth)
        return loss

    def inference_edge_sparse(
        self, batch_processed_data: SparseDataBatch
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred, e_pred = self.model.forward(
            x=batch_processed_data.node_feature, 
            e=batch_processed_data.edge_feature, 
            edge_index=batch_processed_data.edge_index
        )
        del x_pred
        
        # heatmap
        e_pred_softmax = e_pred.softmax(dim=-1)
        e_heatmap = e_pred_softmax[:, 1]
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, batch_processed_data.ground_truth)
            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_edge_dense(
        self, batch_processed_data: DenseDataBatch
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # Inference
        x_pred, e_pred = self.model.forward(
            x=batch_processed_data.node_feature, 
            e=batch_processed_data.graph,
            edge_index=None
        )
        del x_pred
        
        # Get heatmap
        e_pred_softmax = e_pred.softmax(dim=1)
        e_heatmap = e_pred_softmax[:, 1, :, :]
        
        # Return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, batch_processed_data.ground_truth)
            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_node_sparse(
        self, batch_processed_data: SparseDataBatch
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # Inference
        x_pred, e_pred = self.model.forward(
            x=batch_processed_data.node_feature, 
            e=batch_processed_data.edge_feature, 
            edge_index=batch_processed_data.edge_index
        )
        del e_pred
        
        # Get heatmap
        x_pred_softmax = x_pred.softmax(-1)
        x_heatmap = x_pred_softmax[:, 1]
        
        # Return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(x_pred, batch_processed_data.ground_truth)
            return loss, x_heatmap
        elif self.env.mode == "solve":
            return x_heatmap
        else:
            raise ValueError()
    
    def download_weight(self, weight_path: str):
        file_name = os.path.basename(weight_path)
        if file_name not in SUPPORTS:
            raise ValueError(f"Unsupported weight file: {file_name}")
        
        # Download weight
        pull_file_from_huggingface(
            repo_id="ML4CO/ML4CO-Bench-101",
            repo_type="model",
            filename=f"gnn4co/{file_name}",
            save_path=weight_path
        )