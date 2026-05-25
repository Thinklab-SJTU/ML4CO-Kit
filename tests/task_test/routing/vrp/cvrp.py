r"""
CVRP Task Tester.
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


import os
import pathlib
from ml4co_kit import CVRPTask, env_checker
from tests.task_test.base import TaskTesterBase


if env_checker.check_cp39_or_later():
    pickle_files_list = [
        pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl"),
        pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp500_uniform_task.pkl"),
        pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp50_o_uniform_task.pkl"),
        pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp100_o_uniform_task.pkl"),
    ]
else:
    pickle_files_list = [
        pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl"),
        pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp500_uniform_task.pkl"),
    ] 


class CVRPTaskTester(TaskTesterBase):
    def __init__(self):
        super(CVRPTaskTester, self).__init__(
            test_task_class=CVRPTask,
            pickle_files_list=pickle_files_list
        )
        
    def _test_other_rw_methods(self):
        pass
        
    def _test_render(self):
        # Read data
        task = CVRPTask()
        task.from_pickle("test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl")
        task.sol = task.ref_sol
        
        # Render (problem)
        tmp_path = self._make_tmp_file()
        task.render(save_path=pathlib.Path(tmp_path + ".png"), with_sol=False)
        
        # Render (solution)
        task.render(save_path=pathlib.Path(tmp_path + "_sol.png"), with_sol=True)
        
        # Clean up
        os.remove(tmp_path + ".png")
        os.remove(tmp_path + "_sol.png")