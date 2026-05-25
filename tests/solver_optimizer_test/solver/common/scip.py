r"""
SCIP Solver Tester.
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


from ml4co_kit import TASK_TYPE, SCIPSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class SCIPSolverTester(SolverTesterBase):
    def __init__(self):
        super(SCIPSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=SCIPSolver,
            test_task_type_list=[
                TASK_TYPE.MAXRETPO,
                TASK_TYPE.MINVARPO,
                TASK_TYPE.MOPO,
            ],
            test_args_list=[
                {"scip_time_limit": 2.0}, # MaxRetPO
                {"scip_time_limit": 2.0}, # MinVarPO
                {"scip_time_limit": 2.0}, # MOPO
            ],
            exclude_test_files_list=[
                [], # MaxRetPO
                [], # MinVarPO
                [], # MOPO
            ]
        )
        
    def pre_test(self):
        pass