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


import pathlib
from ml4co_kit import TASK_TYPE, GurobiSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class GurobiSolverTester(SolverTesterBase):
    def __init__(self):
        super(GurobiSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=GurobiSolver,
            test_task_type_list=[
                TASK_TYPE.ATSP, 
                TASK_TYPE.CVRP, 
                TASK_TYPE.TSP, 
                TASK_TYPE.MCL, 
                TASK_TYPE.MCUT, 
                TASK_TYPE.MIS, 
                TASK_TYPE.MVC,
            ],
            test_args_list=[
                {}, # ATSP
                {}, # CVRP
                {}, # TSP
                {}, # MCl
                {}, # MCut
                {}, # MIS
                {}, # MVC
            ],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/atsp/task/atsp500_uniform_task.pkl")
                ],  # ATSP
                [
                    pathlib.Path("test_dataset/cvrp/task/cvrp500_uniform_task.pkl")
                ],  # CVRP
                [
                    pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl")
                ],  # TSP
                [], # MCl
                [], # MCut
                [], # MIS
                [], # MVC
            ]
        )
        
    def pre_test(self):
        pass