r"""
OP Wrapper Tester.
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


import shutil
import pathlib
from ml4co_kit import SATAWrapper, SATAGenerator, PySATSolver
from tests.wrapper_test.base import WrapperTesterBase


class SATAWrapperTester(WrapperTesterBase):
    def __init__(self):
        super(SATAWrapperTester, self).__init__(
            test_wrapper_class=SATAWrapper,
            generator=SATAGenerator(),
            solver=PySATSolver(pysat_solver_name="cadical195"),
            pickle_files_list=[
                pathlib.Path("test_dataset/sat/sata/wrapper/sata_ca-small_16ins.pkl"),
            ],
            txt_files_list=[
                pathlib.Path("test_dataset/sat/sata/wrapper/sata_ca-small_16ins.txt"),
            ],
        )
        
    def _test_other_rw_methods(self):

        ###############################################################
        #    Test-1: Solve the CNF data and evaluate the solution     #
        ###############################################################

        # 1.1 Read data from CNF data
        wrapper = SATAWrapper()
        wrapper.from_cnf_folder(
            cnf_folder_path=pathlib.Path("test_dataset/sat/sata/satlib")
        )
        
        # 1.2 Using PySATSolver to solve
        solver = PySATSolver(pysat_solver_name="cadical195")
        wrapper.solve(solver=solver, show_time=True)

        # 1.3 Evaluate the solution
        eval_result = wrapper.evaluate()
        print(f"SATLIB for SATA: {eval_result}")
        
        ###############################################################
        #      Test-2: Transfer data in TXT format to CNF format      #
        ###############################################################

        # 2.1 Read txt data and transfer it to CNF format
        wrapper = SATAWrapper()
        txt_path = pathlib.Path("test_dataset/sat/sata/wrapper/sata_ca-small_16ins.txt")
        wrapper.from_txt(txt_path)
        tmp_name = self._make_tmp_file()
        tmp_cnf_folder_path = pathlib.Path(tmp_name + "_cnf")
        wrapper.to_cnf_folder(cnf_folder_path=tmp_cnf_folder_path)        
        
        # 2.2 Clean up
        shutil.rmtree(tmp_cnf_folder_path)