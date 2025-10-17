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


import os
import shutil
import pathlib
from ml4co_kit import MISWrapper, MISGenerator, LcDegreeSolver, get_md5
from tests.wrapper_test.base import WrapperTesterBase


class MISWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(MISWrapperTester, self).__init__(
            test_wrapper_class=MISWrapper,
            generator=MISGenerator(),
            solver=LcDegreeSolver(),
            pickle_files_list=[
                pathlib.Path("test_dataset/mis/wrapper/mis_er-700-800_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_satlib_no-weighted_4ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/mis/wrapper/mis_er-700-800_no-weighted_4ins.txt"),
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_no-weighted_4ins.txt"),
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.txt"),
                pathlib.Path("test_dataset/mis/wrapper/mis_satlib_no-weighted_4ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):
        
        ###############################################################
        #     Test-1: Solve the gpickle-result data and evaluate      #
        ###############################################################
        
        # 1.1 Read gpickle-result data using ``from_gpickle_result_folder``
        wrapper = MISWrapper()
        wrapper.from_gpickle_result_folder(
            graph_folder_path=pathlib.Path("test_dataset/mis/gpickle_result/instance"),
            result_foler_path=pathlib.Path("test_dataset/mis/gpickle_result/solution"),
            ref=True,
            overwrite=True
        )
        
        # 1.2 Using LcDegreeSolver to solve
        solver = LcDegreeSolver()
        wrapper.solve(solver=solver, show_time=True)
        
        # 1.3 Evaluate the solution
        eval_result = wrapper.evaluate_w_gap()
        print(f"Gpickle-result for MIS: {eval_result}")
        
        ###############################################################
        #           Test-2: Compare two gpickle-result data           #
        ###############################################################
        
        # 2.1 Transfer data to gpickle-result format
        wrapper.to_gpickle_result_folder(
            graph_folder_path=pathlib.Path("tmp/tmp_mis_instance"),
            result_foler_path=pathlib.Path("tmp/tmp_mis_solution"),
        )
        
        # 2.2 Read two gpickle-result data
        wrapper = MISWrapper()
        wrapper.from_gpickle_result_folder(
            graph_folder_path=pathlib.Path("tmp/tmp_mis_instance"),
            result_foler_path=pathlib.Path("tmp/tmp_mis_solution"),
            ref=False,
            overwrite=True,
        )
        wrapper.from_gpickle_result_folder(
            result_foler_path=pathlib.Path("test_dataset/mis/gpickle_result/solution"),
            ref=True,
            overwrite=False,
        )
        
        # 2.3 Evaluate the solution
        new_eval_result = wrapper.evaluate_w_gap()
        print(f"Gpickle-result for MIS: {new_eval_result}")
        
        # 2.4 Compare two evaluation results
        if eval_result != new_eval_result:
            raise ValueError("Inconsistent evaluation results.")
        
        # 2.4 Clean up
        shutil.rmtree(pathlib.Path("tmp/tmp_mis_instance"))
        shutil.rmtree(pathlib.Path("tmp/tmp_mis_solution"))
        
        
        ###############################################################
        #    Test-3: Transfer TXT format to gpickle-result format     #
        ###############################################################
        
        # 3.1 Read txt data and transfer it to gpickle-result format
        txt_path = pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.txt")
        wrapper = MISWrapper()
        wrapper.from_txt(file_path=txt_path,)
        wrapper.to_gpickle_result_folder(
            graph_folder_path=pathlib.Path("tmp/tmp_mis_instance"),
            result_foler_path=pathlib.Path("tmp/tmp_mis_solution"),
        )
        
        # 3.2 Verify conversion consistency
        wrapper = MISWrapper()
        wrapper.from_gpickle_result_folder(
            graph_folder_path=pathlib.Path("tmp/tmp_mis_instance"),
            result_foler_path=pathlib.Path("tmp/tmp_mis_solution"),
            ref=False,
            overwrite=True,
        )
        tmp_txt_path = pathlib.Path("tmp/tmp_mis.txt")
        wrapper.to_txt(tmp_txt_path)
        if get_md5(txt_path) != get_md5(tmp_txt_path):
            raise ValueError("Inconsistent txt data.")
        
        # 3.3 Clean up
        shutil.rmtree(pathlib.Path("tmp/tmp_mis_instance"))
        shutil.rmtree(pathlib.Path("tmp/tmp_mis_solution"))
        os.remove(tmp_txt_path)