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
from ml4co_kit import TASK_TYPE, GreedySolver, TwoOptOptimizer
from ml4co_kit.extension.gnn4co import (
    GNN4COModel, GNN4COEnv, GNNEncoder, TSPGNNEncoder
)
from tests.solver_optimizer_test.base import SolverTesterBase


# Test on ATSP-50 (dense)
gnn4atsp50_model = GNN4COModel(
    env=GNN4COEnv(task="ATSP", mode="solve", sparse_factor=0, device="cpu"),
    encoder=GNNEncoder(task="ATSP", sparse=False, block_layers=[2,4,4,2]),
    weight_path="weights/gnn4co_atsp50_dense.pt"
)

# Test on TSP-50 (dense)
gnn4tsp50_model = GNN4COModel(
    env=GNN4COEnv(task="TSP", mode="solve", sparse_factor=0, device="cpu"),
    encoder=TSPGNNEncoder(sparse=False),
    weight_path="weights/gnn4co_tsp50_dense.pt"
)

# Test on TSP-500 (sparse)
gnn4tsp500_model = GNN4COModel(
    env=GNN4COEnv(task="TSP", mode="solve", sparse_factor=50, device="cpu"),
    encoder=TSPGNNEncoder(sparse=True),
    weight_path="weights/gnn4co_tsp500_sparse.pt"
)

# Optimizer
optimizer = TwoOptOptimizer()


class TwoOptOptimizerTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(TwoOptOptimizerTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=GreedySolver,
            test_task_type_list=[
                TASK_TYPE.ATSP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.TSP, 
            ],
            test_args_list=[
                # ATSP-50 (dense)
                {"model": gnn4atsp50_model, "device": device, "optimizer": optimizer},
                # TSP-50 (dense)
                {"model": gnn4tsp50_model, "device": device, "optimizer": optimizer},
                # TSP-500 (sparse)
                {"model": gnn4tsp500_model, "device": device, "optimizer": optimizer},
            ],
            exclude_test_files_list=[
                [], # ATSP-50 (dense)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp500_uniform_single.pkl"), 
                ],  # TSP-50 (dense)
                [
                    pathlib.Path("test_dataset/tsp/task/tsp50_cluster_single.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_single.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_uniform_single.pkl"), 
                ],  # TSP-500 (sparse)
            ]
        )
        
    def pre_test(self):
        pass