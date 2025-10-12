r"""
NeuroLKH Solver Tester.
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
from ml4co_kit import TASK_TYPE, NeuroLKHSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class NeuroLKHSolverTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(NeuroLKHSolverTester, self).__init__(
            mode_list=["batch_solve"],
            test_solver_class=NeuroLKHSolver,
            test_task_type_list=[TASK_TYPE.TSP],
            test_args_list=[
                {"neurolkh_device": device}
            ],
            exclude_test_files_list=[
                [pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl")]
            ]
        )
        
    def pre_test(self):
        solver = NeuroLKHSolver()
        solver.install()