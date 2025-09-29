r"""
MIS Wrapper Tester.
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
from ml4co_kit import MISWrapper, MISGenerator, KaMISSolver
from tests.wrapper_test.base import WrapperTesterBase


class MISWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(MISWrapperTester, self).__init__(
            test_wrapper_class=MISWrapper,
            generator=MISGenerator(),
            solver=KaMISSolver(kamis_time_limit=1.0),
            pickle_files_list=[
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_16ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_er-700-800_4ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_satlib_4ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_16ins.txt"),
                pathlib.Path("test_dataset/mis/wrapper/mis_er-700-800_4ins.txt"),
                pathlib.Path("test_dataset/mis/wrapper/mis_satlib_4ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):
        pass