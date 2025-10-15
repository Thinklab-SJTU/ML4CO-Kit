r"""
SAT Generator Test Module.
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
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_folder)

from .base import GenTesterBase
from ml4co_kit.generator.logic.sat import SATGenerator
from ml4co_kit.task.logic.sat import SATTask


class SATGenTester(GenTesterBase):
    def __init__(self):
        super(SATGenTester, self).__init__(
            task_cls=SATTask,
            generator_cls=SATGenerator
        )

    def test(self):
        print("Testing SAT Generator...")
        
        # Test different distributions
        distributions = ['RANDOM', 'UNIFORM_RANDOM', 'PLANTED', 'PHASE_TRANSITION', 'INDUSTRIAL']
        
        for distribution in distributions:
            print(f"  Testing {distribution} distribution...")
            
            # Generate instance
            instance = self.generator.generate(
                num_vars=10,
                num_clauses=30,
                distribution=distribution,
                seed=42
            )
            
            # Basic validation
            assert instance.num_vars == 10
            assert instance.get_num_clauses() == 30
            assert len(instance.clauses) == 30
            
            # Check clause validity
            for clause in instance.clauses:
                for literal in clause:
                    assert abs(literal) <= 10  # Variable index within range
                    assert literal != 0        # No zero literals
            
            print(f"    ✅ {distribution} distribution test passed")
        
        # Test reproducibility
        print("  Testing reproducibility...")
        instance1 = self.generator.generate(num_vars=5, num_clauses=15, distribution='RANDOM', seed=123)
        instance2 = self.generator.generate(num_vars=5, num_clauses=15, distribution='RANDOM', seed=123)
        
        assert instance1.clauses == instance2.clauses
        print("    ✅ Reproducibility test passed")
        
        print("✅ SAT Generator test completed successfully!")