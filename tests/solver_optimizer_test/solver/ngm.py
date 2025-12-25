r"""
NGM Solver Tester.
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


from ml4co_kit import TASK_TYPE, NGMSolver
from tests.solver_optimizer_test.base import SolverTesterBase


class NGMSolverTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(NGMSolverTester, self).__init__(
            mode_list=["batch_solve"],
            test_solver_class=NGMSolver,
            test_task_type_list=[TASK_TYPE.GM],
            test_args_list=[
                {
                    "gnn_channels": (16, 16, 16),
                    "sk_emb": 1,
                    "sk_max_iter": 50,
                    "pretrain" : 'voc',
                    "network": None,
                    "device": device,
                },  # GM
            ],
            exclude_test_files_list=[
                [],
            ] # GM
        )
        
    def pre_test(self):
        pass