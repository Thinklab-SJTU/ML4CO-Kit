r"""
Global Prediction Degree Solver Tester.
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
from ml4co_kit import TASK_TYPE, LcDegreeSolver
from tests.solver_test.base import SolverTesterBase


class LcDegreeSolverTester(SolverTesterBase):
    def __init__(self):
        super(LcDegreeSolverTester, self).__init__(
            test_solver_class=LcDegreeSolver,
            test_files_list=[
                pathlib.Path("test_dataset/mis/mis_single_task.pkl"),
                pathlib.Path("test_dataset/mvc/mvc_single_task.pkl"),
                pathlib.Path("test_dataset/mcl/mcl_single_task.pkl"),
                pathlib.Path("test_dataset/mcut/mcut_single_task.pkl"),
             ],
            test_tasks_list=[
                TASK_TYPE.MIS, 
                TASK_TYPE.MVC, 
                TASK_TYPE.MCL, 
                TASK_TYPE.MCUT,
            ],
            test_args_list=[
                # MIS
                {},
                # MVC
                {},
                # MCL
                {},
                # MCUT
                {}
            ]
        )
        
    def pre_test(self):
        pass