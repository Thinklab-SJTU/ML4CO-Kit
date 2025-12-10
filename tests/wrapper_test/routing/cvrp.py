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
        
        ###############################################################
        # Test-1: Solve the VRPLIB data (X) and evaluate the solution #
        ###############################################################
        
        # 1.1 Read Real-World (X) VRPLIB data using ``from_vrplib_folder``
        wrapper = CVRPWrapper()
        wrapper.from_vrplib_folder(
            vrp_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/problem_1"),
            sol_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/solution_1"),
            ref=True,
            overwrite=True,
            normalize=True
        )
            
        # 1.2 Using HGSSolver to solve
        solver = HGSSolver()
        wrapper.solve(solver=solver, show_time=True)
        
        # 1.3 Evaluate the solution under the normalized data
        eval_result = wrapper.evaluate_w_gap()
        print(f"VRPLIB (X) for CVRP (normalize=True): {eval_result}")
        
        # 1.4 Using ``overwrite`` to evaluate solution under the original data
        wrapper.from_vrplib_folder(
            vrp_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/problem_1"),
            sol_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/solution_1"),
            ref=True,
            overwrite=False,
            normalize=False
        )
        eval_result = wrapper.evaluate_w_gap()
        print(f"VRPLIB (X) for CVRP (normalize=False): {eval_result}")
        
        
        ###############################################################
        # Test-2: Solve the VRPLIB data (A) and evaluate the solution #
        ###############################################################
        
        # 2.1 Read Real-World (A) VRPLIB data using ``from_vrplib_folder``
        wrapper.from_vrplib_folder(
            vrp_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/problem_2"),
            sol_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/solution_2"),
            ref=True,
            overwrite=True,
            normalize=True
        )
        
        # 2.2 Using HGSSolver to solve
        solver = HGSSolver()
        wrapper.solve(solver=solver, show_time=True)
        
        # 2.3 Evaluate the solution under the normalized data
        eval_result = wrapper.evaluate_w_gap()
        print(f"VRPLIB (A) for CVRP (normalize=True): {eval_result}")
        
        # 2.4 Using ``overwrite`` to evaluate solution under the original data
        wrapper.from_vrplib_folder(
            vrp_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/problem_2"),
            sol_folder_path=pathlib.Path("test_dataset/cvrp/vrplib/solution_2"),
            ref=True,
            overwrite=False,
            normalize=False
        )
        eval_result = wrapper.evaluate_w_gap()
        print(f"VRPLIB (A) for CVRP (normalize=False): {eval_result}")
        
        ###############################################################
        #    Test-3: Transfer data in TXT format to VRPLIB format     #
        ###############################################################
        
        # 3.1 Read pickle data and transfer it to VRPLIB format
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
        
        # 3.2 Verify conversion consistency
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
        
        # 3.3 Clean up
        shutil.rmtree(tmp_vrp_folder_path)
        shutil.rmtree(tmp_sol_folder_path)
        os.remove(tmp_txt_path)