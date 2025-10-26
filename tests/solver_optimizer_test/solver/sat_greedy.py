r"""
SAT Greedy Solver Tester.
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


from ml4co_kit import TASK_TYPE, GreedySolver
from ml4co_kit.extension.gnn4co import (
    GNN4COModel, GNN4COEnv, GNNEncoder
)
from tests.solver_optimizer_test.base import SolverTesterBase


# Create a dummy model for SAT greedy solver (uses heuristic, not neural network)
# This is required by the GreedySolver interface but not actually used for SAT
gnn4sat_model = GNN4COModel(
    env=GNN4COEnv(task="SAT", mode="solve", sparse_factor=1, device="cpu"),
    encoder=GNNEncoder(task="SAT", sparse=True, block_layers=[2,4,4,2]),
    weight_path=None  # No weights needed for greedy heuristic
)


class SATGreedySolverTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(SATGreedySolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=GreedySolver,
            test_task_type_list=[TASK_TYPE.SAT],
            test_args_list=[
                {"model": gnn4sat_model, "device": device}  # SAT with greedy heuristic
            ],
            exclude_test_files_list=[
                []  # SAT
            ]
        )
        
    def pre_test(self):
        pass