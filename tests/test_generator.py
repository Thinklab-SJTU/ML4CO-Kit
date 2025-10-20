r"""
Test Generator Module.
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
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)


# Routing Problems
from tests.generator_test import (
    ATSPGenTester, 
    CVRPGenTester, 
    OPGenTester, 
    PCTSPGenTester, 
    SPCTSPGenTester, 
    TSPGenTester
)


# Graph Problems
from tests.generator_test import (
    MClGenTester, 
    MCutGenTester, 
    MISGenTester, 
    MVCGenTester
)

# Portfolio Problems
from tests.generator_test import (
    MinVarPOGenTester, 
    MaxRetPOGenTester, 
    MOPOGenTester
)


# Test Generator
def test_generator():
    # Routing Problems
    ATSPGenTester().test()
    CVRPGenTester().test()
    OPGenTester().test()
    PCTSPGenTester().test()
    SPCTSPGenTester().test()
    TSPGenTester().test()

    # Graph Problems
    MClGenTester().test()
    MCutGenTester().test()
    MISGenTester().test()
    MVCGenTester().test()

    # Portfolio Problems
    MinVarPOGenTester().test()
    MaxRetPOGenTester().test()
    MOPOGenTester().test()


# Main
if __name__ == "__main__":
    test_generator()