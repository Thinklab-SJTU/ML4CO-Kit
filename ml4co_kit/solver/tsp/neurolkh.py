r"""
NeuroLKH Solver for solving TSPs.

LKH is a heuristic algorithm that uses k-opt move strategies
to find approximate optimal solutions to problems.

We follow https://github.com/liangxinedu/NeuroLKH
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


import os
import torch
import pathlib
import numpy as np
from typing import Union
from multiprocessing import Pool
from joblib import Parallel, delayed
from ml4co_kit.solver.tsp.base import TSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer
from ml4co_kit.utils.file_utils import pull_file_from_huggingface
from ml4co_kit.solver.tsp.pyneurolkh.encoder import SparseGCNEncoder
from ml4co_kit.solver.tsp.pyneurolkh.sparser import neurolkh_sparser
from ml4co_kit.solver.tsp.pyneurolkh.wrapper import neurolkh_wrapper, alkh_wrapper


class TSPNeuroLKHSolver(TSPSolver):
    r"""
    Solve TSPs using LKH solver.

    :param scale, int, the scale factor for coordinates.
    :param lkh_max_trials, int, the maximum number of trials for the LKH solver.
    :param lkh_path, pathlib.Path, the path of the LKH solver.
    :param lkh_runs, int, the number of runs for the LKH solver.
    :param lkh_seed, int, the random number seed for the LKH solver.
    :param lkh_special, boolean, whether to solve in a special way.
    """
    def __init__(
        self,
        scale: int = 1e5,
        lkh_tree_cands_num: int = 10,
        lkh_search_cands_num: int = 5,
        lkh_max_trials: int = 500,
        lkh_path: pathlib.Path = "LKH",
        lkh_runs: int = 1,
        lkh_seed: int = 1234,
        lkh_special: bool = False,
        lkh_initial_period: int = 15,
        use_nn: bool = True,
        sparse_factor: int = 20,
        neurolkh_device: str = "cpu"
    ):
        super(TSPNeuroLKHSolver, self).__init__(
            solver_type=SOLVER_TYPE.NEUROLKH, scale=scale
        )
        
        # lkh params
        self.lkh_tree_cands_num = lkh_tree_cands_num
        self.lkh_search_cands_num = lkh_search_cands_num
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_runs = lkh_runs
        self.lkh_seed = lkh_seed
        self.lkh_special = lkh_special
        self.lkh_initial_period = lkh_initial_period
        
        # use nn
        self.use_nn = use_nn
        if self.use_nn:
            # check pretrain file
            pyneurolkh_path = pathlib.Path(__file__).parent
            pretrain_path = pyneurolkh_path / "pyneurolkh/pretrained_file/neurolkh.pt"
            if not os.path.exists(pretrain_path):
                pull_file_from_huggingface(
                    repo_id="ML4CO/ML4CO-Kit", 
                    repo_type="model", 
                    filename="neurolkh/neurolkh_original.pt", 
                    save_path=pretrain_path
                )
                
            # neurolkh encoder
            self.neurolkh_device = neurolkh_device
            self.sparse_factor = sparse_factor
            encoder = SparseGCNEncoder(sparse_factor=sparse_factor)
            encoder.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
            self.encoder = encoder.to(neurolkh_device)

    def _solve_with_nn(
        self, nodes_coord: np.ndarray, penalty: np.ndarray,
        heatmap: np.ndarray, full_edge_index: np.ndarray,
    ) -> list:
        r"""
        Solve a single TSP instance
        """
        tour = neurolkh_wrapper(
            points=nodes_coord,
            penalty=penalty,
            heatmap=heatmap,
            full_edge_index=full_edge_index,
            sparse_factor=self.sparse_factor,
            lkh_tree_cands_num=self.lkh_tree_cands_num,
            lkh_search_cands_num=self.lkh_search_cands_num,
            lkh_scale=self.scale,
            lkh_max_trials=self.lkh_max_trials,
            lkh_path=self.lkh_path,
            lkh_runs=self.lkh_runs,
            lkh_seed=self.lkh_seed,
            lkh_special=self.lkh_special,
        )
        return tour
    
    def _solve_without_nn(self, nodes_coord: np.ndarray) -> list:
        r"""
        Solve a single TSP instance
        """
        tour = alkh_wrapper(
            points=nodes_coord,
            lkh_tree_cands_num=self.lkh_tree_cands_num,
            lkh_search_cands_num=self.lkh_search_cands_num,
            lkh_scale=self.scale,
            lkh_max_trials=self.lkh_max_trials,
            lkh_path=self.lkh_path,
            lkh_runs=self.lkh_runs,
            lkh_seed=self.lkh_seed,
            lkh_special=self.lkh_special,
            lkh_initial_period=self.lkh_initial_period
        )
        return tour
    
    def solve(
        self,
        points: Union[np.ndarray, list] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        batch_size: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        r"""
        :param points: np.ndarray, the coordinates of nodes. If given, the points 
            originally stored in the solver will be replaced.
        :param norm: boolean, the normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        """
        if self.use_nn:
            return self.solve_with_nn(
                points=points,
                norm=norm,
                normalize=normalize,
                num_threads=num_threads,
                batch_size=batch_size,
                show_time=show_time
            )
        else:
            return self.solve_without_nn(
                points=points,
                norm=norm,
                normalize=normalize,
                num_threads=num_threads,
                show_time=show_time
            )
    
    def solve_with_nn(
        self,
        points: Union[np.ndarray, list] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        batch_size: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        
        # preparation
        self.from_data(points=points, norm=norm, normalize=normalize)
        if self.norm != "EUC_2D":
            raise ValueError(
                "The current version only supports Euler-2D distance"
            )
        timer = Timer(apply=show_time)
        timer.start()
        
        # nodes number
        nodes_num = self.points.shape[1]
        
        # sparse
        edge_index_list = list()
        graph_list = list()
        inverse_edge_index_list = list()
        full_edge_index_list = list()
        for _points in self.points:
            # call the sparser
            edge_index, graph, inverse_edge_index = neurolkh_sparser(
                points=_points, sparse_factor=self.sparse_factor
            ) # (V, K), (V, K), (V, K)
            
            # full_edge_index
            full_edge_index_0 = torch.arange(nodes_num).reshape((-1, 1))
            full_edge_index_0 = full_edge_index_0.repeat(1, self.sparse_factor).reshape(-1)
            full_edge_index_1 = torch.from_numpy(edge_index.reshape(-1))
            full_edge_index = torch.stack([full_edge_index_0, full_edge_index_1], dim=0)  
            full_edge_index = full_edge_index.numpy() # (2, V*K)
            
            # numpy -> tensor
            th_graph = torch.from_numpy(graph).reshape(-1) # (V*K,)
            th_edge_index = torch.from_numpy(edge_index).reshape(-1) # (V*K,)
            th_inverse_edge_index = torch.from_numpy(inverse_edge_index).reshape(-1) # (V*K,)
            
            # add to lists
            edge_index_list.append(th_edge_index)
            graph_list.append(th_graph)
            inverse_edge_index_list.append(th_inverse_edge_index)
            full_edge_index_list.append(full_edge_index)
        
        # cat
        th_all_points = torch.from_numpy(self.points).to(self.neurolkh_device) # (S, V, 2)
        th_all_edge_index = torch.stack(edge_index_list, dim=0).to(self.neurolkh_device) # (S, V*K)
        th_all_graph = torch.stack(graph_list, dim=0).to(self.neurolkh_device) # (S, V*K)
        th_all_inverse_edge_index = torch.stack(inverse_edge_index_list, dim=0).to(self.neurolkh_device) # (S, V*K)

        # bacth
        samples = th_all_points.shape[0]
        assert samples % batch_size == 0, "The batch size must be divisible by the number of samples."
        batch_num = samples // batch_size
        batch_points = th_all_points.reshape(batch_num, batch_size, nodes_num, 2)
        batch_edge_index = th_all_edge_index.reshape(batch_num, batch_size, -1)
        batch_graph = th_all_graph.reshape(batch_num, batch_size, -1)
        batch_inverse_edge_index = th_all_inverse_edge_index.reshape(batch_num, batch_size, -1)

        # call encoder to get the penalty and heatmap
        penalty_list = list() 
        heatmap_list = list()
        
        with torch.no_grad():
            for idx in iterative_execution(range, batch_num, "Call SparseGCNEncoder", show_time):
                # call encoder
                penalty, alpha = self.encoder.forward(
                    x=batch_points[idx],
                    graph=batch_graph[idx], 
                    edge_index=batch_edge_index[idx], 
                    inverse_edge_index=batch_inverse_edge_index[idx]
                )
                
                # format
                penalty = penalty.squeeze(dim=-1) # (B, V)
                penalty = penalty.detach().cpu().numpy()
                alpha = alpha.permute(0, 2, 1) # (B, 2, V*K)
                heatmap = alpha.softmax(dim=1)[:, 1, :] # (B, V*K)
                heatmap = heatmap.detach().cpu().numpy()
                
                # add to lists
                penalty_list.append(penalty)
                heatmap_list.append(heatmap)
        penalty = np.concatenate(penalty_list, axis=0)
        heatmap = np.concatenate(heatmap_list, axis=0)
        
        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                tours.append(
                    self._solve_with_nn(
                        nodes_coord=self.points[idx],
                        penalty=penalty[idx],
                        heatmap=heatmap[idx],
                        full_edge_index=full_edge_index_list[idx]
                    )
                )
        else:
            # bacth
            batch_points = self.points.reshape(-1, num_threads, p_shape[-2], p_shape[-1])
            batch_penalty = penalty.reshape(-1, num_threads, nodes_num)
            batch_heatmap = heatmap.reshape(-1, num_threads, nodes_num*self.sparse_factor)
            batch_full_edge_index = np.stack(full_edge_index_list, axis=0)
            batch_full_edge_index = batch_full_edge_index.reshape(-1, num_threads, 2, nodes_num*self.sparse_factor)

            # parallel solving
            for idx in iterative_execution(
                range, num_points // num_threads, self.solve_msg, show_time
            ):
                cur_tours = Parallel(n_jobs=num_threads)(
                    delayed(self._solve_with_nn)(
                        batch_points[idx][i],
                        batch_penalty[idx][i],
                        batch_heatmap[idx][i],
                        batch_full_edge_index[idx][i]
                    )
                    for i in range(num_threads)
                )
                for tour in cur_tours:
                    tours.append(tour)

        # format
        tours = np.array(tours)
        zeros = np.zeros((tours.shape[0], 1))
        tours = np.append(tours, zeros, axis=1).astype(np.int32)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()

        return tours

    def solve_without_nn(
        self,
        points: Union[np.ndarray, list] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        
        # preparation
        self.from_data(points=points, norm=norm, normalize=normalize)
        if self.norm != "EUC_2D":
            raise ValueError(
                "The current version only supports Euler-2D distance"
            )
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                tours.append(self._solve_without_nn(self.points[idx]))
        else:
            batch_points = self.points.reshape(-1, num_threads, p_shape[-2], p_shape[-1])
            for idx in iterative_execution(
                range, num_points // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.map(
                        self._solve_without_nn,
                        [
                            batch_points[idx][inner_idx]
                            for inner_idx in range(num_threads)
                        ],
                    )
                for tour in cur_tours:
                    tours.append(tour)
        
        # format
        tours = np.array(tours)
        zeros = np.zeros((tours.shape[0], 1))
        tours = np.append(tours, zeros, axis=1).astype(np.int32)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return self.tours

    def __str__(self) -> str:
        return "TSPNeuroLKHSolver"