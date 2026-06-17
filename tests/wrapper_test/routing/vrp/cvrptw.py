r"""
CVRPTW Wrapper Tester.
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
from ml4co_kit import CVRPTWWrapper, CVRPTWGenerator, PyVRPSolver
from tests.wrapper_test.base import WrapperTesterBase


class CVRPTWWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(CVRPTWWrapperTester, self).__init__(
            test_wrapper_class=CVRPTWWrapper,
            generator=CVRPTWGenerator(),
            solver=PyVRPSolver(),
            pickle_files_list=[
                pathlib.Path("test_dataset/routing/vrp/cvrptw/wrapper/cvrptw50_uniform_16ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/routing/vrp/cvrptw/wrapper/cvrptw50_uniform_16ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):
        pass