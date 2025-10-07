r"""
CVRP Wrapper Tester.
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
from ml4co_kit import CVRPWrapper, CVRPGenerator, HGSSolver, get_md5
from tests.wrapper_test.base import WrapperTesterBase


class CVRPWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(CVRPWrapperTester, self).__init__(
            test_wrapper_class=CVRPWrapper,
            generator=CVRPGenerator(),
            solver=HGSSolver(hgs_time_limit=5.0),
            pickle_files_list=[
                pathlib.Path("test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/cvrp/wrapper/cvrp500_uniform_4ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.txt"),
                pathlib.Path("test_dataset/cvrp/wrapper/cvrp500_uniform_4ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):
        self._test_vrplib_folder()
        
    def _test_vrplib_folder(self):
        # Step1-1: Read Real-World (X) VRPLIB data using ``from_vrplib_folder``
        wrapper = CVRPWrapper()
        for normalize in [True, False]:
            wrapper.from_vrplib_folder(
                vrp_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/problem_1"),
                sol_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/solution_1"),
                ref=True,
                overwrite=True,
                normalize=normalize
            )
            
            # Using HGSSolver to solve and evaluate
            solver = HGSSolver() if normalize else HGSSolver(hgs_scale=1)
            wrapper.solve(solver=solver, show_time=True)
            eval_result = wrapper.evaluate_w_gap()
            print(f"VRPLIB (X) for CVRP (normalize={normalize}): {eval_result}")
        
        # Step1-2: Read Real-World (A) VRPLIB data using ``from_vrplib_folder``
        for normalize in [True, False]:
            wrapper.from_vrplib_folder(
                vrp_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/problem_2"),
                sol_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/solution_2"),
                ref=True,
                overwrite=True,
                normalize=normalize
            )
            
            # Using HGSSolver to solve and evaluate
            solver = HGSSolver() if normalize else HGSSolver(hgs_scale=1)
            wrapper.solve(solver=solver, show_time=True)
            eval_result = wrapper.evaluate_w_gap()
            print(f"VRPLIB (A) for CVRP (normalize={normalize}): {eval_result}")
        
        # Step2: Read pickle data and transfer it to VRPLIB format
        txt_path = pathlib.Path("test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.txt")
        pkl_path = pathlib.Path("test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.pkl")
        wrapper.from_pickle(pkl_path)
        wrapper.swap_sol_and_ref_sol()
        tmp_name = self._make_tmp_file()
        tmp_vrp_folder_path = pathlib.Path(tmp_name + "_vrp")
        tmp_sol_folder_path = pathlib.Path(tmp_name + "_sol")
        wrapper.to_vrplib_folder(
            vrp_folder_path=tmp_vrp_folder_path,
            sol_folder_path=tmp_sol_folder_path,
        )
        
        # Step3: Read VRPLIB data and transfer it to pickle format to check the correctness
        wrapper.from_vrplib_folder(
            vrp_folder_path=tmp_vrp_folder_path,
            sol_folder_path=tmp_sol_folder_path,
            ref=False,
            overwrite=True,
        )
        tmp_txt_path = pathlib.Path(tmp_name + ".txt")
        wrapper.to_txt(tmp_txt_path)
        if get_md5(txt_path) != get_md5(tmp_txt_path):
            raise ValueError("Inconsistent txt data.")
        
        # Clean up
        shutil.rmtree(tmp_vrp_folder_path)
        shutil.rmtree(tmp_sol_folder_path)
        os.remove(tmp_txt_path)