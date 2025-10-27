r"""
SAT Greedy Solver Tester.

This tester uses the GreedySolver with heuristic-based approach for SAT.
SAT greedy solving doesn't require a GNN model (uses DPLL and unit propagation).
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


from ml4co_kit import TASK_TYPE
from ml4co_kit.solver import GreedySolver
from tests.solver_optimizer_test.base import SolverTesterBase


class SATGreedySolverTester(SolverTesterBase):
    """
    SAT Greedy Solver Tester.
    
    Uses heuristic-based greedy approach (DPLL, unit propagation, frequency-based 
    variable selection) without requiring a neural network model.
    """
    
    def __init__(self, device: str = "cpu"):
        super(SATGreedySolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=GreedySolver,
            test_task_type_list=[TASK_TYPE.SAT],
            test_args_list=[
                {
                    "model": None,  # SAT doesn't need a model
                    "device": device
                }
            ],
            exclude_test_files_list=[[]]
        )
        
    def pre_test(self):
        """Pre-test setup."""
        pass