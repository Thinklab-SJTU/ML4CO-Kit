r"""
Fast 2-opt Optimizer Tester.
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
from ml4co_kit import *
from tests.solver_optimizer_test.base import SolverTesterBase



class FastTwoOptOptimizerTester(SolverTesterBase):
    def __init__(self):
        super(FastTwoOptOptimizerTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=NearestSolver,
            test_task_type_list=[TASK_TYPE.TSP],
            test_args_list=[{"optimizer": FastTwoOptOptimizer()}],
            exclude_test_files_list=[
                [
                    pathlib.Path("test_dataset/routing/tsp/task/tsp50_cluster_task.pkl"),
                    pathlib.Path("test_dataset/routing/tsp/task/tsp50_gaussian_task.pkl"),
                    pathlib.Path("test_dataset/routing/tsp/task/tsp50_uniform_task.pkl"), 
                    pathlib.Path("test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.pkl"), 
                ],
            ],
            info="Fast 2-opt Optimizer"
        )
        
    def pre_test(self):
        pass