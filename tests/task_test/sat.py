r"""
SAT Task Test Module.
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
import sys
import numpy as np
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_folder)

from .base import TaskTesterBase
from ml4co_kit.task.logic.sat import SATTask


class SATTaskTester(TaskTesterBase):
    def __init__(self):
        super(SATTaskTester, self).__init__(task_cls=SATTask)

    def test(self):
        print("Testing SAT Task...")
        
        # Create a test SAT instance
        # Formula: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
        clauses = [
            [1, 2],      # x1 OR x2
            [-1, 3],     # NOT x1 OR x3
            [-2, -3]     # NOT x2 OR NOT x3
        ]
        
        # Test from_data method
        print("  Testing from_data method...")
        task = self.task_cls()
        task.from_data(clauses=clauses, num_vars=3)
        
        assert task.num_vars == 3
        assert task.get_num_clauses() == 3
        assert task.clauses == clauses
        print("    ✅ from_data test passed")
        
        # Test solution evaluation
        print("  Testing solution evaluation...")
        
        # Test satisfying assignment: x1=True, x2=False, x3=True
        satisfying_assignment = np.array([1, 0, 1])
        task.set_assignment(satisfying_assignment)
        
        satisfied_clauses = task.evaluate(satisfying_assignment)
        assert satisfied_clauses == 3  # All clauses should be satisfied
        assert task.is_satisfiable(satisfying_assignment)
        
        # Test unsatisfying assignment: x1=False, x2=False, x3=False
        unsatisfying_assignment = np.array([0, 0, 0])
        satisfied_clauses_2 = task.evaluate(unsatisfying_assignment)
        assert satisfied_clauses_2 < 3  # Not all clauses satisfied
        assert not task.is_satisfiable(unsatisfying_assignment)
        
        print("    ✅ Solution evaluation test passed")
        
        # Test copy functionality
        print("  Testing copy functionality...")
        copied_task = task.copy()
        assert copied_task.num_vars == task.num_vars
        assert copied_task.clauses == task.clauses
        assert np.array_equal(copied_task.assignment, task.assignment)
        print("    ✅ Copy test passed")
        
        # Test constraint checking
        print("  Testing constraint checking...")
        assert task.check_constraints(satisfying_assignment)
        assert not task.check_constraints(unsatisfying_assignment)
        print("    ✅ Constraint checking test passed")
        
        print("✅ SAT Task test completed successfully!")