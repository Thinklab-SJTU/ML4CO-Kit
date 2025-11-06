r"""
ISCO Solver Tester.
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
import numpy as np
from ml4co_kit import TASK_TYPE, ISCOSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class ISCOSolverTester(SolverTesterBase):
    def __init__(self):
        super(ISCOSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=ISCOSolver,
            test_task_type_list=[
                TASK_TYPE.MCL, TASK_TYPE.MCUT, TASK_TYPE.MIS, TASK_TYPE.MVC,
            ],
            test_args_list=[
                # MCl
                {
                    "isco_init_type": "uniform",
                    "isco_tau": 0.5,
                    "isco_mu_init": 5.0,
                    "isco_g_func": lambda r: np.sqrt(r),
                    "isco_adapt_mu": True,
                    "isco_target_accept_rate": 0.574,
                    "isco_alpha": 0.3,
                    "isco_beta": 1.02,
                    "isco_iterations": 10000,
                    "isco_seed": 1234
                }, 
                # MCut
                {
                    "isco_tau": 0.5,
                    "isco_mu_init": 5.0,
                    "isco_g_func": lambda r: np.sqrt(r),
                    "isco_adapt_mu": True,
                    "isco_target_accept_rate": 0.574,
                    "isco_iterations": 10000,
                    "isco_seed": 1234
                },
                # MIS
                {
                    "isco_init_type": "uniform",
                    "isco_tau": 0.5,
                    "isco_mu_init": 5.0,
                    "isco_g_func": lambda r: np.sqrt(r),
                    "isco_adapt_mu": True,
                    "isco_target_accept_rate": 0.574,
                    "isco_alpha": 0.3,
                    "isco_beta": 1.02,
                    "isco_iterations": 10000,
                    "isco_seed": 1234
                },
                # MVC
                {
                    "isco_init_type": "uniform",
                    "isco_tau": 0.5,
                    "isco_mu_init": 5.0,
                    "isco_g_func": lambda r: np.sqrt(r),
                    "isco_adapt_mu": True,
                    "isco_target_accept_rate": 0.574,
                    "isco_alpha": 0.3,
                    "isco_beta": 1.02,
                    "isco_iterations": 50000,
                    "isco_seed": 1234
                }
            ],
            exclude_test_files_list=[
                [], 
                [], 
                [pathlib.Path("test_dataset/mis/task/mis_satlib_no-weighted_task.pkl")],
                [pathlib.Path("test_dataset/mvc/task/mvc_rb-large_no-weighted_task.pkl")]
            ]
        )
        
    def pre_test(self):
        pass