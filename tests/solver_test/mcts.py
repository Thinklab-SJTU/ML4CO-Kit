r"""
MCTS Solver Tester.
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
from ml4co_kit import TASK_TYPE, MCTSSolver
from ml4co_kit.extension.gnn4co import GNN4COModel, GNN4COEnv, TSPGNNEncoder
from tests.solver_test.base import SolverTesterBase


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


class MCTSSolverTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(MCTSSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=MCTSSolver,
            test_task_type_list=[
                TASK_TYPE.TSP, TASK_TYPE.TSP
            ],
            test_args_list=[
                {
                    "model": gnn4tsp50_model,
                    "mcts_time_limit": 0.05,
                    "mcts_max_depth": 10,
                    "mcts_type_2opt": 1,
                    "mcts_max_iterations_2opt": 5000
                },
                {
                    "model": gnn4tsp500_model,
                    "mcts_time_limit": 1.0,
                    "mcts_max_depth": 10,
                    "mcts_type_2opt": 2,
                    "mcts_max_iterations_2opt": 5000
                }   
            ],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl"), 
                ],
                [
                    pathlib.Path("test_dataset/tsp/task/tsp50_cluster_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_task.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_uniform_task.pkl"), 
                ],
            ]
        )
        
    def pre_test(self):
        pass