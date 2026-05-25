r"""
CVRPBLTW Task Tester.
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
from ml4co_kit import CVRPBLTWTask
from tests.task_test.base import TaskTesterBase


class CVRPBLTWTaskTester(TaskTesterBase):
    def __init__(self):
        super(CVRPBLTWTaskTester, self).__init__(
            test_task_class=CVRPBLTWTask,
            pickle_files_list=[
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpbltw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpbltw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpbltw50_o_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpbltw100_o_uniform_task.pkl"),
            ],
        )
        
    def _test_other_rw_methods(self):
        pass
        
    def _test_render(self):
        pass