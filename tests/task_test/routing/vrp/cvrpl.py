r"""
CVRPL Task Tester.
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
from ml4co_kit import CVRPLTask
from tests.task_test.base import TaskTesterBase


class CVRPLTaskTester(TaskTesterBase):
    def __init__(self):
        super(CVRPLTaskTester, self).__init__(
            test_task_class=CVRPLTask,
            pickle_files_list=[
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/cvrpl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/cvrpl100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/ovrpl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/ovrpl100_uniform_task.pkl"),
            ],
        )
        
    def _test_other_rw_methods(self):
        pass
        
    def _test_render(self):
        pass