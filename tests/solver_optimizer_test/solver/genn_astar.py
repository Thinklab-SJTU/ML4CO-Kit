r"""
GENN_AStar Solver Tester.
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
from ml4co_kit import TASK_TYPE, GennAStarSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class GennAStarSolverTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(GennAStarSolverTester, self).__init__(
            mode_list=["batch_solve"],
            test_solver_class=GennAStarSolver,
            test_task_type_list=[TASK_TYPE.GM,], #TASK_TYPE.GED],
            test_args_list=[
                {
                    "channel": 36,
                    "filters_1": 64,
                    "filters_2": 32,
                    "filters_3": 16,
                    "tensor_neurons": 16,
                    "beam_width": 0,
                    "trust_fact": 1,
                    "no_pred_size": 0,
                    "network": None,
                    "pretrain": "AIDS700nef",
                    "device": device,
                },  # GM
                {
                    {
                    "channel": 36,
                    "filters_1": 64,
                    "filters_2": 32,
                    "filters_3": 16,
                    "tensor_neurons": 16,
                    "beam_width": 0,
                    "trust_fact": 1,
                    "no_pred_size": 0,
                    "network": None,
                    "pretrain": "AIDS700nef",
                    "device": device,
                },  # GED
                }
            ],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/gm/wrapper/gm_er_large_uniform_4ins.pkl"),  
                ],
                []
                ]
        )
        
    def pre_test(self):
        pass