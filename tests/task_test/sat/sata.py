r"""
SAT-A Task Tester.
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
import pathlib
from ml4co_kit import SATATask, PySATSolver, get_md5
from tests.task_test.base import TaskTesterBase


class SATATaskTester(TaskTesterBase):
    def __init__(self):
        super(SATATaskTester, self).__init__(
            test_task_class=SATATask,
            pickle_files_list=[
                pathlib.Path("test_dataset/sat/sata/task/sata_ca-small_task.pkl"),
            ],
        )
        
    def _test_other_rw_methods(self):

        ##################################################
        #        Test-1: Read data from CNF file         #
        ##################################################

        # 1.1 Read data from TSPLIB file
        task = SATATask()
        f1_path = pathlib.Path("test_dataset/sat/sata/satlib/CBS_k3_n100_m403_b10_0.cnf")
        task.from_cnf(file_path=f1_path)
        
        # 1.2 Solve the problem
        solver = PySATSolver()
        solver.solve(task)

        # 1.3 Evaluate the solution
        cost = task.evaluate(task.sol)
        print(f"CBS_k3_n100_m403_b10_0: {cost}")
        
        # 1.4 Output data using ``to_cnf``
        tmp_name = self._make_tmp_file()
        tmp_cnf_file_path = pathlib.Path(tmp_name + "_cnf")
        task.to_cnf(file_path=tmp_cnf_file_path)
        
        # Check md5
        f1_md5 = get_md5(f1_path)
        f2_md5 = get_md5(tmp_cnf_file_path)
        if f1_md5 != f2_md5:
            raise ValueError("Inconsistent CNF data.")
        
        # 1.5 Clean up
        os.remove(tmp_cnf_file_path)
    
    def _test_render(self):
        pass