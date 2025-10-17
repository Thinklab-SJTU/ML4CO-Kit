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
import numpy as np
import scipy.sparse
from typing import Any
from torch import Tensor, nn
from typing import Union, Tuple, List
from ml4co_kit.learning.model import BaseModel
from ml4co_kit.utils.file_utils import pull_file_from_huggingface
from ml4co_kit.utils.type_utils import to_numpy, to_tensor
from ..env.env import GNN4COEnv
from .encoder.gnn_encoder import GNNEncoder


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
        
        # load pretrained weights if needed
        if weight_path is not None:
            if not os.path.exists(weight_path):
                self.download_weight(weight_path)
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        self.to(self.env.device)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # set mode
        self.env.mode = phase
        
        # get real data
        """
        task: ATSP or CVRP or MCl or MCut or MIS or MVC or TSP
        if sparse:
            [0] task
            [1] x: (V, C) or (V,) , nodes feature
            [2] e: (E, D) or (E,) , edges feature
            [3] edge_index: (2, E)
            [4] graph_list: graph data
            [5] ground_truth: (E,) or (V,)
            [6] nodes_num_list
            [7] edges_num_list
        else:
            [0] task
            [1] x: (B, V, C) or (B, V), nodes_feature
            [2] graph: (B, V, V)
            [3] ground_truth: (B, V, V) or (B, V)
            [4] nodes_num_list
        """
        if phase == "train":
            # get real train batch data
            batch_size = len(batch)
            batch_data = self.env.generate_train_data(batch_size)
            task = batch_data[0]
            
            # deal with different task
            if task in ["TSP", "ATSP", "CVRP"]:
                if self.env.sparse:
                    loss = self.train_edge_sparse_process(*batch_data)
                else:
                    loss = self.train_edge_dense_process(*batch_data)
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                if self.env.sparse:
                    loss = self.train_node_sparse_process(*batch_data)
                else:
                    loss = self.train_node_dense_process(*batch_data)
            else:
                raise NotImplementedError()
            
        elif phase == "val":
            # get val data
            batch_data = self.env.generate_val_data(batch_idx)
            task = batch_data[0]
            
            # deal with different task
            if task in ["TSP", "ATSP", "CVRP"]:
                if self.env.sparse:
                    loss, heatmap = self.inference_edge_sparse_process(*batch_data)
                else:
                    loss, heatmap = self.inference_edge_dense_process(*batch_data)
                    
            elif task in ["MIS", "MCut", "MCl", "MVC"]:
                if self.env.sparse:
                    loss, heatmap = self.inference_node_sparse_process(*batch_data)
                else:
                    loss, heatmap = self.inference_node_dense_process(*batch_data)
            else:
                raise NotImplementedError()
            
            # decoding
            if self.env.sparse:
                costs_avg = self.decoder.sparse_decode(
                    heatmap, *batch_data, return_cost=True
                )
            else:
                costs_avg = self.decoder.dense_decode(
                    heatmap, *batch_data, return_cost=True
                )
        else:
            raise NotImplementedError()
     
        # log
        metrics = {f"{phase}/loss": loss}
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg})
        for k, v in metrics.items():
            formatted_v = f"{v:.8f}"
            self.log(k, float(formatted_v), prog_bar=True, on_epoch=True, sync_dist=True)
        
        # return
        return loss if phase == "train" else metrics   
        
    def train_edge_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, 
        graph_list: List[Tensor], ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del x_pred
        loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
        return loss
   
    def train_edge_dense_process(
        self, task: str, x: Tensor, graph: Tensor, 
        ground_truth: Tensor, nodes_num_list: list
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=graph, edge_index=None
        )
        del x_pred
        loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
        return loss
    
    def train_node_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, 
        graph_list: List[Tensor], ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list
    ) -> Tensor:
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del e_pred
        loss = nn.CrossEntropyLoss()(x_pred, ground_truth)
        return loss

    def train_node_dense_process(
        self, task: str, x: Tensor, graph: Tensor, 
        ground_truth: Tensor, nodes_num_list: list
    ) -> Tensor:
        raise NotImplementedError()

    def inference_edge_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, 
        graph_list: List[Tensor], ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del x_pred
        
        # heatmap
        e_pred_softmax = e_pred.softmax(dim=-1)
        e_heatmap = e_pred_softmax[:, 1]
        
        # sparse -> dense
        heatmap_list = list()
        edge_begin_idx = 0
        for nodes_num, edges_num in zip(nodes_num_list, edges_num_list):
            edge_end_idx = edge_begin_idx + edges_num
            _heatmap = to_numpy(e_heatmap[edge_begin_idx:edge_end_idx])
            _edge_index = to_numpy(edge_index)
            dense_heatmap = scipy.sparse.coo_matrix(
                arg1=(_heatmap, (_edge_index[0], _edge_index[1])), 
                shape=(nodes_num, nodes_num)
            ).toarray()
            dense_heatmap = (dense_heatmap + dense_heatmap.T) / 2
            dense_heatmap = np.clip(dense_heatmap, a_min=1e-14, a_max=1-1e-14)
            heatmap_list.append(to_tensor(dense_heatmap).to(self.env.device))
            edge_begin_idx = edge_end_idx
        e_heatmap = torch.stack(heatmap_list, 0)
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_edge_dense_process(
        self, task: str, x: Tensor, graph: Tensor, 
        ground_truth: Tensor, nodes_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=graph, edge_index=None
        )
        del x_pred
        
        # heatmap
        e_pred_softmax = e_pred.softmax(dim=1)
        e_heatmap = e_pred_softmax[:, 1, :, :]
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
            return loss, e_heatmap
        elif self.env.mode == "solve":
            return e_heatmap
        else:
            raise ValueError()

    def inference_node_sparse_process(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor, 
        graph_list: List[Tensor], ground_truth: Tensor, 
        nodes_num_list: list, edges_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # inference
        x_pred, e_pred = self.model.forward(
            task=task, x=x, e=e, edge_index=edge_index
        )
        del e_pred
        
        # heatmap
        x_pred_softmax = x_pred.softmax(-1)
        x_heatmap = x_pred_softmax[:, 1]
        
        # return
        if self.env.mode == "val":
            loss = nn.CrossEntropyLoss()(x_pred, ground_truth)
            return loss, x_heatmap
        elif self.env.mode == "solve":
            return x_heatmap
        else:
            raise ValueError()
        
    def inference_node_dense_process(
        self, task: str, x: Tensor, graph: Tensor, 
        ground_truth: Tensor, nodes_num_list: list
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError()
    
    def download_weight(self, weight_path: str):
        file_name = os.path.basename(weight_path)
        if file_name not in SUPPORTS:
            raise ValueError(f"Unsupported weight file: {file_name}")
        
        # download
        pull_file_from_huggingface(
            repo_id="ML4CO/ML4CO-Bench-101",
            repo_type="model",
            filename=f"gnn4co/{file_name}",
            save_path=weight_path
        )