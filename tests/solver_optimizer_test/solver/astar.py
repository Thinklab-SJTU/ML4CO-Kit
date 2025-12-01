r"""
AStar Solver Tester.
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
from ml4co_kit import TASK_TYPE, AStarSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class AStarSolverTester(SolverTesterBase):
    def __init__(self):
        super(AStarSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=AStarSolver,
            test_task_type_list=[
                TASK_TYPE.GM,
                TASK_TYPE.GED,
            ],
            test_args_list=[
                {
                    "beam_width": 0
                }, # GM
                {
                     "beam_width": 0
                } # GED
            ],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/gm/task/gm_er_large_uniform_task.pkl"),
                ],
                [
                ]
                ]
        )
        
    def pre_test(self):
        pass