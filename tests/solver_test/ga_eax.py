r"""
GA_EAX Solver Tester.
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
from ml4co_kit import TASK_TYPE, GAEAXSolver
from tests.solver_test.base import SolverTesterBase


class GAEAXSolverTester(SolverTesterBase):
    def __init__(self):
        super(GAEAXSolverTester, self).__init__(
            test_solver_class=GAEAXSolver,
            test_task_type_list=[TASK_TYPE.TSP, TASK_TYPE.TSP],
            test_args_list=[
                {
                    "use_large_solver": False
                },
                {
                    "use_large_solver": True,
                    "ga_eax_population_num": 300,
                    "ga_eax_offspring_num": 30,
                },
            ],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/tsp/task/tsp500_uniform_single.pkl"), 
                ],
                [
                    pathlib.Path("test_dataset/tsp/task/tsp50_cluster_single.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_single.pkl"),
                    pathlib.Path("test_dataset/tsp/task/tsp50_uniform_single.pkl"), 
                ], 
            ]
        )
        
    def pre_test(self):
        pass