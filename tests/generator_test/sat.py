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
from ml4co_kit.generator.logic.base import LOGIC_TYPE


class SATGenTester(GenTesterBase):
    def __init__(self):
        super(SATGenTester, self).__init__(
            test_gen_class=SATGenerator,
            test_args_list=[
                # Random distribution
                {
                    "distribution_type": LOGIC_TYPE.RANDOM,
                    "num_vars": 10,
                    "num_clauses": 30,
                    "clause_length": 3,
                },
                # Uniform random distribution
                {
                    "distribution_type": LOGIC_TYPE.UNIFORM_RANDOM,
                    "num_vars": 15,
                    "num_clauses": 45,
                    "clause_length": 3,
                },
                # Planted distribution
                {
                    "distribution_type": LOGIC_TYPE.PLANTED,
                    "num_vars": 8,
                    "num_clauses": 24,
                    "clause_length": 3,
                },
                # Phase transition distribution
                {
                    "distribution_type": LOGIC_TYPE.PHASE_TRANSITION,
                    "num_vars": 12,
                    "clause_length": 3,
                },
                # Industrial distribution
                {
                    "distribution_type": LOGIC_TYPE.INDUSTRIAL,
                    "num_vars": 20,
                    "num_clauses": 60,
                    "clause_length": 3,
                },
            ]
        )