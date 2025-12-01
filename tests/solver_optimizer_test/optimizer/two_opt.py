r"""
Greedy Solver Tester.
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


import pathlib
from ml4co_kit import *
from ml4co_kit.extension.gnn4co import (
    GNN4COModel, GNN4COEnv, GNNEncoder, TSPGNNEncoder, GNN4COGreedyDecoder
)
from tests.solver_optimizer_test.base import SolverTesterBase


# Test on ATSP-50 (dense)
gnn4atsp50_model = GNN4COModel(
    env=GNN4COEnv(
        task_type=TASK_TYPE.ATSP, 
        wrapper=ATSPWrapper(), 
        mode="solve", 
        sparse_factor=-1, 
        device="cpu"
    ),
    encoder=GNNEncoder(
        task_type=TASK_TYPE.ATSP, 
        sparse=False, 
        block_layers=[2,4,4,2]
    ),
    decoder=GNN4COGreedyDecoder(sparse_factor=-1),
    weight_path="weights/gnn4co_atsp50_dense.pt"
)

# Test on TSP-50 (dense)
gnn4tsp50_model = GNN4COModel(
    env=GNN4COEnv(
        task_type=TASK_TYPE.TSP, 
        wrapper=TSPWrapper(), 
        mode="solve", 
        sparse_factor=-1, 
        device="cpu"
    ),
    encoder=TSPGNNEncoder(sparse=False),
    decoder=GNN4COGreedyDecoder(sparse_factor=-1),
    weight_path="weights/gnn4co_tsp50_dense.pt"
)

# Test on TSP-500 (sparse)
gnn4tsp500_model = GNN4COModel(
    env=GNN4COEnv(
        task_type=TASK_TYPE.TSP, 
        wrapper=TSPWrapper(), 
        mode="solve", 
        sparse_factor=50, 
        device="cpu"
    ),
    encoder=TSPGNNEncoder(sparse=True),
    decoder=GNN4COGreedyDecoder(sparse_factor=50),
    weight_path="weights/gnn4co_tsp500_sparse.pt"
)

# Optimizers
optimizer_ctypes = TwoOptOptimizer(impl_type=IMPL_TYPE.CTYPES)
optimizer_pybind11_v1 = TwoOptOptimizer(impl_type=IMPL_TYPE.PYBIND11, type_2opt=1)
optimizer_pybind11_v2 = TwoOptOptimizer(impl_type=IMPL_TYPE.PYBIND11, type_2opt=2)
optimizer_torch = TwoOptOptimizer(impl_type=IMPL_TYPE.TORCH)


class TwoOptOptimizerTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(TwoOptOptimizerTester, self).__init__(
            mode_list=["solve", "batch_solve_parallel"],
            test_solver_class=GNN4COSolver,
            test_task_type_list=[
                TASK_TYPE.ATSP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.TSP, 
            ],
            test_args_list=[
                # ATSP-50 (dense)
                {"model": gnn4atsp50_model, "device": device, "optimizer": optimizer_ctypes},
                # TSP-50 (dense)
                {"model": gnn4tsp50_model, "device": device, "optimizer": optimizer_pybind11_v1},
                {"model": gnn4tsp50_model, "device": device, "optimizer": optimizer_pybind11_v2},
                {"model": gnn4tsp50_model, "device": device, "optimizer": optimizer_torch},
                # TSP-500 (sparse)
                {"model": gnn4tsp500_model, "device": device, "optimizer": optimizer_pybind11_v1},
                {"model": gnn4tsp500_model, "device": device, "optimizer": optimizer_pybind11_v2},
                {"model": gnn4tsp500_model, "device": device, "optimizer": optimizer_torch},
            ],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/atsp/wrapper/atsp500_uniform_4ins.pkl"),
                ], # ATSP-50 (dense)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl"), 
                    pathlib.Path("test_dataset/tsp/wrapper/tsp500_uniform_4ins.pkl"), 
                ],  # TSP-50 (dense)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl"), 
                    pathlib.Path("test_dataset/tsp/wrapper/tsp500_uniform_4ins.pkl"), 
                ],  # TSP-50 (dense)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl"), 
                    pathlib.Path("test_dataset/tsp/wrapper/tsp500_uniform_4ins.pkl"), 
                ],  # TSP-50 (dense)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp50_cluster_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_uniform_task.pkl"), 
                    pathlib.Path("test_dataset/tsp/wrapper/tsp50_uniform_16ins.pkl"), 
                ],  # TSP-500 (sparse)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp50_cluster_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_uniform_task.pkl"), 
                    pathlib.Path("test_dataset/tsp/wrapper/tsp50_uniform_16ins.pkl"), 
                ],  # TSP-500 (sparse)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp50_cluster_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_uniform_task.pkl"), 
                    pathlib.Path("test_dataset/tsp/wrapper/tsp50_uniform_16ins.pkl"), 
                ],  # TSP-500 (sparse)
            ],
            info="Two-opt Optimizer"
        )
        
    def pre_test(self):
        pass