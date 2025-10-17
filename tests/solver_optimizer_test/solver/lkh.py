r"""
LKH Solver Tester.
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


from ml4co_kit import TASK_TYPE, LKHSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class LKHSolverTester(SolverTesterBase):
    def __init__(self):
        super(LKHSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=LKHSolver,
            test_task_type_list=[
                TASK_TYPE.TSP, TASK_TYPE.ATSP, TASK_TYPE.CVRP
            ],
            test_args_list=[
                {}, {}, {"lkh_special": True},
            ],
            exclude_test_files_list=[[], [], []]
        )
        
    def pre_test(self):
        solver = LKHSolver()
        solver.install()