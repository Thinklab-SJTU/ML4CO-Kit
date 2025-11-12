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
from tests.solver_test.base import SolverTesterBase


class AStarSolverTester(SolverTesterBase):
    def __init__(self):
        super(AStarSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=AStarSolver,
            test_task_type_list=[
                TASK_TYPE.GM
            ],
            test_args_list=[
                {
                    "beam_width": 0
                } 
            ],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/gm/task/gm_er_uniform_iso_task.pkl"),
                    pathlib.Path("test_dataset/gm/task/gm_er_uniform_ind_task.pkl"),  
                    pathlib.Path("test_dataset/gm/task/gm_er_gaussian_iso_task.pkl"),
                    pathlib.Path("test_dataset/gm/task/gm_er_gaussian_ind_task.pkl"),
                ]
                ]
        )
        
    def pre_test(self):
        pass