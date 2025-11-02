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
import pathlib
import numpy as np
from ml4co_kit.task.logic.sat import SATTask
from tests.task_test.base import TaskTesterBase


class SATTaskTester(TaskTesterBase):
    def __init__(self):
        # SAT doesn't have pickle files yet, so use empty list
        super(SATTaskTester, self).__init__(
            test_task_class=SATTask,
            pickle_files_list=[
                # Add SAT pickle files here when available
            ],
        )

    def _test_other_rw_methods(self):
        """Test SAT-specific read/write methods."""
        
        ##################################################
        #       Test-1: Test from_data method            #
        ##################################################
        
        # Create a test SAT instance
        # Formula: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
        clauses = [
            [1, 2],      # x1 OR x2
            [-1, 3],     # NOT x1 OR x3
            [-2, -3]     # NOT x2 OR NOT x3
        ]
        
        task = SATTask()
        task.from_data(clauses=clauses)
        
        # Test satisfying assignment: x1=True, x2=False, x3=True
        satisfying_assignment = np.array([1, 0, 1])
        satisfied_clauses = task.evaluate(satisfying_assignment)
        assert satisfied_clauses == 3, f"Expected 3 satisfied clauses, got {satisfied_clauses}"
        
        # Test unsatisfying assignment: x1=False, x2=False, x3=False
        unsatisfying_assignment = np.array([0, 0, 0])
        satisfied_clauses_2 = task.evaluate(unsatisfying_assignment)
        assert satisfied_clauses_2 < 3, f"Expected < 3 satisfied clauses, got {satisfied_clauses_2}"
        
        print(f"SATTask basic test: {satisfied_clauses} clauses satisfied with valid assignment")
        
        ##################################################
        #       Test-2: Test DIMACS format support       #
        ##################################################
        
        # Create a temporary DIMACS file
        tmp_dimacs_path = self._make_tmp_file() + ".cnf"
        
        # Write DIMACS format
        with open(tmp_dimacs_path, 'w') as f:
            f.write("c This is a test SAT instance\n")
            f.write("p cnf 3 3\n")
            f.write("1 2 0\n")
            f.write("-1 3 0\n")
            f.write("-2 -3 0\n")
        
        # Test loading from DIMACS
        dimacs_task = SATTask()
        dimacs_task.from_dimacs(pathlib.Path(tmp_dimacs_path))
        
        # Verify the loaded data
        assert dimacs_task.num_vars == 3
        assert len(dimacs_task.cnf) == 3
        assert dimacs_task.cnf == clauses
        
        # Clean up
        os.remove(tmp_dimacs_path)
        
        print("SATTask DIMACS format test passed")

    def _test_render(self):
        """Test SAT visualization (if implemented)."""
        # SAT visualization is optional and complex
        # For now, just pass
        pass