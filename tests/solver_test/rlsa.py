r"""
RLSA Solver Tester.
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


from ml4co_kit import TASK_TYPE, RLSASolver
from tests.solver_test.base import SolverTesterBase


class RLSASolverTester(SolverTesterBase):
    def __init__(self):
        super(RLSASolverTester, self).__init__(
            test_solver_class=RLSASolver,
            test_task_type_list=[
                TASK_TYPE.MCL, TASK_TYPE.MCUT, TASK_TYPE.MIS, TASK_TYPE.MVC,
            ],
            test_args_list=[
                # MCL
                {
                    "rlsa_kth_dim": "both",
                    "rlsa_d": 2,
                    "rlsa_k": 1000,
                    "rlsa_t": 1000,
                    "rlsa_alpha": 0.3,
                    "rlsa_beta": 1.02,
                    "rlsa_device": "cuda",
                    "rlsa_seed": 1234
                }, 
                # MCUT
                {
                    "rlsa_kth_dim": 0,
                    "rlsa_d": 2,
                    "rlsa_k": 1000,
                    "rlsa_t": 1000,
                    "rlsa_alpha": 0.3,
                    "rlsa_beta": 1.02,
                    "rlsa_device": "cuda",
                    "rlsa_seed": 1234
                },
                # MIS
                {
                    "rlsa_kth_dim": "both",
                    "rlsa_d": 2,
                    "rlsa_k": 1000,
                    "rlsa_t": 1000,
                    "rlsa_alpha": 0.3,
                    "rlsa_beta": 1.02,
                    "rlsa_device": "cuda",
                    "rlsa_seed": 1234
                },
                # MVC
                {
                    "rlsa_kth_dim": "both",
                    "rlsa_d": 2,
                    "rlsa_k": 1000,
                    "rlsa_t": 1000,
                    "rlsa_alpha": 0.3,
                    "rlsa_beta": 1.02,
                    "rlsa_device": "cuda",
                    "rlsa_seed": 1234
                }
            ],
            exclude_test_files_list=[[], [], [], []]
        )
        
    def pre_test(self):
        pass