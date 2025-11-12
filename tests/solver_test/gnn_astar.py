r"""
GNN_AStar Solver Tester.
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
from ml4co_kit import TASK_TYPE, GNN_AStarSolver
from tests.solver_test.base import SolverTesterBase


class GNN_AStarSolverTester(SolverTesterBase):
    def __init__(self):
        super(GNN_AStarSolverTester, self).__init__(
            mode_list=["batch_solve"],
            test_solver_class=GNN_AStarSolver,
            test_task_type_list=[TASK_TYPE.GM],
            test_args_list=[
                {
                    "channel": None,
                    "filters_1": 64,
                    "filters_2": 32,
                    "filters_3": 16,
                    "tensor_neurons": 16,
                    "beam_width": 0,
                    "trust_fact": 1,
                    "no_pred_size": 0,
                    "network": None,
                    "pretrain": "AIDS700nef",
                    "device": "cpu"
                },  # GM
            ],
            exclude_test_files_list=[[]]
        )
        
    def pre_test(self):
        pass