r"""
MCut Wrapper Tester.
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
import shutil
import pathlib
from ml4co_kit import (
    MCutWrapper, MCutGenerator, LcDegreeSolver, GRAPH_TYPE, get_md5
)
from tests.wrapper_test.base import WrapperTesterBase


class MCutWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(MCutWrapperTester, self).__init__(
            test_wrapper_class=MCutWrapper,
            generator=MCutGenerator(
                distribution_type=GRAPH_TYPE.BA,
            ),
            solver=LcDegreeSolver(),
            pickle_files_list=[
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-large_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-small_uniform-weighted_4ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-large_no-weighted_4ins.txt"),
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-small_no-weighted_4ins.txt"),
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-small_uniform-weighted_4ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):
        
        ###############################################################
        #     Test-1: Transfer TXT format to gpickle-result format     #
        ###############################################################
        
        # 1.1 Read txt data and transfer it to gpickle-result format
        txt_path = pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-small_uniform-weighted_4ins.txt")
        wrapper = MCutWrapper()
        wrapper.from_txt(file_path=txt_path)
        wrapper.to_gpickle_result_folder(
            graph_folder_path=pathlib.Path("tmp/tmp_mcut_instance"),
            result_foler_path=pathlib.Path("tmp/tmp_mcut_solution"),
        )
        
        # 3.2 Verify conversion consistency
        wrapper = MCutWrapper()
        wrapper.from_gpickle_result_folder(
            graph_folder_path=pathlib.Path("tmp/tmp_mcut_instance"),
            result_foler_path=pathlib.Path("tmp/tmp_mcut_solution"),
            ref=False,
            overwrite=True,
        )
        tmp_txt_path = pathlib.Path("tmp/tmp_mcut.txt")
        wrapper.to_txt(tmp_txt_path)
        if get_md5(txt_path) != get_md5(tmp_txt_path):
            raise ValueError("Inconsistent txt data.")
        
        # 3.3 Clean up
        shutil.rmtree(pathlib.Path("tmp/tmp_mcut_instance"))
        shutil.rmtree(pathlib.Path("tmp/tmp_mcut_solution"))
        os.remove(tmp_txt_path)