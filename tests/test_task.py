r"""
Test Task Module.
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
from tests.task_test import (
    ATSPTaskTester, 
    CVRPTaskTester, 
    TSPTaskTester, 
    PCTSPTaskTester
)    


# Graph Problems
from tests.task_test import (
    MClTaskTester, 
    MISTaskTester,
    MVCTaskTester,
    MCutTaskTester
)

# Portfolio Problems
from tests.task_test import (
    MaxRetPOTaskTester,
    MinVarPOTaskTester,
    MOPOTaskTester
)


# Test Task
def test_task():
    # Routing Problems
    ATSPTaskTester().test()
    CVRPTaskTester().test()
    PCTSPTaskTester().test()
    TSPTaskTester().test()
    
    # Graph Problems
    MClTaskTester().test()
    MCutTaskTester().test()
    MISTaskTester().test()
    MVCTaskTester().test()

    # Portfolio Problems
    MaxRetPOTaskTester().test()
    MinVarPOTaskTester().test()
    MOPOTaskTester().test()
    

# Main
if __name__ == "__main__":
    test_task()