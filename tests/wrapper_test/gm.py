r"""
GM Wrapper Tester.
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
from ml4co_kit import GMWrapper, GMGenerator, RRWMSolver
from tests.wrapper_test.base import WrapperTesterBase


class GMWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(GMWrapperTester, self).__init__(
            test_wrapper_class=GMWrapper,
            generator=GMGenerator(),
            solver=RRWMSolver(),
            pickle_files_list=[
                pathlib.Path("test_dataset/gm/wrapper/gm_er_uniform_iso_4ins.pkl"),
                pathlib.Path("test_dataset/gm/wrapper/gm_er_gaussian_iso_4ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/gm/wrapper/gm_er_uniform_iso_4ins.txt"),
                pathlib.Path("test_dataset/gm/wrapper/gm_er_gaussian_iso_4ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):
        pass