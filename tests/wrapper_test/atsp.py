r"""
ATSP Wrapper Tester.
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
from ml4co_kit import ATSPWrapper, ATSPGenerator, LKHSolver
from tests.wrapper_test.base import WrapperTesterBase


class ATSPWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(ATSPWrapperTester, self).__init__(
            test_wrapper_class=ATSPWrapper,
            generator=ATSPGenerator(),
            solver=LKHSolver(),
            pickle_files_list=[
                pathlib.Path("test_dataset/atsp/wrapper/atsp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/atsp/wrapper/atsp500_uniform_4ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/atsp/wrapper/atsp50_uniform_16ins.txt"),
                pathlib.Path("test_dataset/atsp/wrapper/atsp500_uniform_4ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):
        pass