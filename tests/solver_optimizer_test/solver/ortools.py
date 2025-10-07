r"""
Gurobi Solver Tester.
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


from ml4co_kit import TASK_TYPE, ORSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class ORSolverTester(SolverTesterBase):
    def __init__(self):
        super(ORSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=ORSolver,
            test_task_type_list=[
                TASK_TYPE.ATSP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.MCL, 
                TASK_TYPE.MIS, 
                TASK_TYPE.MVC
            ],
            test_args_list=[
                {
                    "ortools_time_limit": 1
                },  # ATSP
                {
                    "ortools_time_limit": 1
                },  # TSP
                {}, # MCl
                {}, # MIS
                {}, # MVC
            ],
            exclude_test_files_list=[
                [], # ATSP
                [], # TSP
                [], # MCl
                [], # MIS
                [], # MVC
            ]
        )
        
    def pre_test(self):
        pass