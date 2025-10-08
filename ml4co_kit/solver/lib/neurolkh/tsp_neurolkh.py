r"""
LKH Algorithm for TSP
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import torch
import pathlib
from typing import List
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.utils.file_utils import download
from ml4co_kit.solver.lib.neurolkh.pyneurolkh.encoder import SparseGCNEncoder
from ml4co_kit.solver.lib.neurolkh.pyneurolkh.sparser import neurolkh_sparser
from ml4co_kit.solver.lib.neurolkh.pyneurolkh.wrapper import neurolkh_wrapper


def batch_tsp_neurolkh(
    batch_task_data: List[TSPTask],
    lkh_scale: int = 1e6,
    lkh_max_trials: int = 500,
    lkh_path: pathlib.Path = "LKH",
    lkh_runs: int = 1,
    lkh_seed: int = 1234,
    lkh_special: bool = False,
    neurolkh_device: str = "cpu",
    neurolkh_tree_cands_num: int = 10,
    neurolkh_search_cands_num: int = 5,
    neurolkh_initial_period: int = 15,
    neurolkh_sparse_factor: int = 20,
    encoder: SparseGCNEncoder = None,
):

    # Check Model Pretrained File
    root_path = pathlib.Path(__file__).parent
    neurolkh_pretrained_path = root_path / "tsp_neurolkh.pt"
    download_link = f"https://huggingface.co/ML4CO/ML4CO-Kit/resolve/main/neurolkh/tsp_neurolkh.pt"
    download(file_path=neurolkh_pretrained_path, url=download_link)
    
    # Load Model
    encoder = SparseGCNEncoder(sparse_factor=neurolkh_sparse_factor)
    encoder.load_state_dict(torch.load(neurolkh_pretrained_path, map_location="cpu"))
    encoder = encoder.to(neurolkh_device)
    
    # Preparation (data)
    points_list = list()
    edge_index_list = list()
    graph_list = list()
    inverse_edge_index_list = list()
    full_edge_index_list = list()
    
    for task_data in batch_task_data:
        points = task_data.points
        nodes_num = points.shape[0]
        edge_index, graph, inverse_edge_index = neurolkh_sparser(
            points=points, sparse_factor=neurolkh_sparse_factor
        ) # (V, K), (V, K), (V, K)
        
        # full_edge_index
        full_edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
        full_edge_index_0 = full_edge_index_0.repeat(1, neurolkh_sparse_factor).reshape(-1)
        full_edge_index_1 = torch.from_numpy(edge_index.reshape(-1))
        full_edge_index = torch.stack([full_edge_index_0, full_edge_index_1], dim=0)  
        full_edge_index = full_edge_index.numpy() # (2, V*K)
        
        # numpy -> tensor
        th_points = torch.from_numpy(points) # (V, 2)
        th_graph = torch.from_numpy(graph).reshape(-1) # (V*K,)
        th_edge_index = torch.from_numpy(edge_index).reshape(-1) # (V*K,)
        th_inverse_edge_index = torch.from_numpy(inverse_edge_index).reshape(-1) # (V*K,)
        
        # add to lists
        points_list.append(th_points)
        edge_index_list.append(th_edge_index)
        graph_list.append(th_graph)
        inverse_edge_index_list.append(th_inverse_edge_index)
        full_edge_index_list.append(full_edge_index)

    # Stack and move to device
    th_all_points = torch.stack(points_list).to(neurolkh_device) # (B, V, 2)
    th_all_edge_index = torch.stack(edge_index_list, dim=0).to(neurolkh_device) # (B, V*K)
    th_all_graph = torch.stack(graph_list, dim=0).to(neurolkh_device) # (B, V*K)
    th_all_inverse_edge_index = torch.stack(inverse_edge_index_list, dim=0).to(neurolkh_device) # (B, V*K)

    # Call encoder
    penalty, alpha = encoder.forward(
        x=th_all_points,
        graph=th_all_graph, 
        edge_index=th_all_edge_index, 
        inverse_edge_index=th_all_inverse_edge_index
    )
    
    # Format
    penalty = penalty.squeeze(dim=-1) # (B, V)
    penalty = penalty.detach().cpu().numpy()
    alpha = alpha.permute(0, 2, 1) # (B, 2, V*K)
    heatmap = alpha.softmax(dim=1)[:, 1, :] # (B, V*K)
    heatmap = heatmap.detach().cpu().numpy()

    # Call neurolkh_wrapper
    for idx, task_data in enumerate(batch_task_data):
        sol = neurolkh_wrapper(
            points=task_data.points,
            penalty=penalty[idx],
            heatmap=heatmap[idx],
            full_edge_index=full_edge_index_list[idx],
            lkh_scale=lkh_scale,
            lkh_max_trials=lkh_max_trials,
            lkh_path=lkh_path,
            lkh_runs=lkh_runs,
            lkh_seed=lkh_seed,
            lkh_special=lkh_special,
            sparse_factor=neurolkh_sparse_factor,
            lkh_tree_cands_num=neurolkh_tree_cands_num,
            lkh_search_cands_num=neurolkh_search_cands_num,
            lkh_initial_period=neurolkh_initial_period
        )
        task_data.from_data(sol=sol, ref=False)