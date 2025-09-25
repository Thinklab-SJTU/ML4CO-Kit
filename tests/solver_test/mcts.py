r"""
MCTS Solver Tester.
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


from ml4co_kit import TASK_TYPE, MCTSSolver
from tests.solver_test.base import SolverTesterBase


class MCTSSolverTester(SolverTesterBase):
    def __init__(self):
        super(MCTSSolverTester, self).__init__(
            test_solver_class=MCTSSolver,
            test_tasks_list=[TASK_TYPE.TSP],
            test_args_list=[
                {
                    "model": None,
                    "mcts_time_limit": 1.0,
                    "mcts_max_depth": 10,
                    "mcts_type_2opt": 1,
                    "mcts_max_iterations_2opt": 5000
                }
            ]
        )