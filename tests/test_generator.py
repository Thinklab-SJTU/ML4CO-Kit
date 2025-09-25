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
from generator_test import (
    ATSPGenTester, CVRPGenTester, TSPGenTester, 
    PCTSPGenTester, SPCTSPGenTester, OPGenTester,
    MCutGenTester, MClGenTester, MISGenTester, MVCGenTester
)


if __name__ == "__main__":
    ATSPGenTester().test()
    CVRPGenTester().test()
    TSPGenTester().test()
    MCutGenTester().test()
    MClGenTester().test()
    MISGenTester().test()
    MVCGenTester().test()
    PCTSPGenTester().test()
    SPCTSPGenTester().test()
    OPGenTester().test()