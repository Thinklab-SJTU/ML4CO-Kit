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


import os
import shutil
import pathlib
from ml4co_kit import ATSPWrapper, ATSPGenerator, LKHSolver, get_md5
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
        
        ###############################################################
        #   Test-1: Solve the TSPLIB data and evaluate the solution   #
        ###############################################################
        
        # 1.1 Read Real-World TSPLIB data using ``from_tsplib_folder``
        wrapper = ATSPWrapper()
        wrapper.from_tsplib_folder(
            atsp_folder_path=pathlib.Path("test_dataset/atsp/tsplib/problem"),
            tour_folder_path=pathlib.Path("test_dataset/atsp/tsplib/solution"),
            ref=True,
            overwrite=True,
            normalize=True
        )
    
        # 1.2 Using LKHSolver to solve and evaluate
        solver = LKHSolver()
        wrapper.solve(solver=solver, show_time=True)
        
        # 1.3 Evaluate the solution under the normalized data
        eval_result = wrapper.evaluate_w_gap()
        print(f"TSPLIB for ATSP (normalize=True): {eval_result}")
        
        # 1.4 Using ``overwrite`` to evaluate solution under the original data
        wrapper.from_tsplib_folder(
            atsp_folder_path=pathlib.Path("test_dataset/atsp/tsplib/problem"),
            tour_folder_path=pathlib.Path("test_dataset/atsp/tsplib/solution"),
            ref=True,
            overwrite=False,
            normalize=False
        )
        eval_result = wrapper.evaluate_w_gap()
        print(f"TSPLIB for ATSP (normalize=False): {eval_result}")
        
        ###############################################################
        #    Test-2: Transfer data in TXT format to TSPLIB format     #
        ###############################################################
        
        # 2.1 Read pickle data and transfer it to TSPLIB format
        txt_path = pathlib.Path("test_dataset/atsp/wrapper/atsp50_uniform_16ins.txt")
        pkl_path = pathlib.Path("test_dataset/atsp/wrapper/atsp50_uniform_16ins.pkl")
        wrapper.from_pickle(pkl_path)
        wrapper.swap_sol_and_ref_sol()
        tmp_name = self._make_tmp_file()
        tmp_atsp_folder_path = pathlib.Path(tmp_name + "_atsp")
        tmp_tour_folder_path = pathlib.Path(tmp_name + "_tour")
        wrapper.to_tsplib_folder(
            atsp_folder_path=tmp_atsp_folder_path,
            tour_folder_path=tmp_tour_folder_path,
        )

        # 2.2 Read TSPLIB data and transfer it to pickle format to check the correctness
        wrapper.from_tsplib_folder(
            atsp_folder_path=tmp_atsp_folder_path,
            tour_folder_path=tmp_tour_folder_path,
            ref=False,
            overwrite=True,
        )
        tmp_txt_path = pathlib.Path(tmp_name + ".txt")
        wrapper.to_txt(tmp_txt_path)
        if get_md5(txt_path) != get_md5(tmp_txt_path):
            raise ValueError("Inconsistent txt data.")
        
        # 2.3 Clean up
        shutil.rmtree(tmp_atsp_folder_path)
        shutil.rmtree(tmp_tour_folder_path)
        os.remove(tmp_txt_path)