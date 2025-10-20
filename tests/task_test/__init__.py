r"""
Task Test Module.
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


# Base Class
from .base import TaskTesterBase

# Routing Problems
from .atsp import ATSPTaskTester
from .cvrp import CVRPTaskTester
from .tsp import TSPTaskTester
from .pctsp import PCTSPTaskTester

# Graph Problems
from .mcl import MClTaskTester
from .mcut import MCutTaskTester
from .mis import MISTaskTester
from .mvc import MVCTaskTester

# Portfolio Problems
from .maxretpo import MaxRetPOTaskTester
from .minvarpo import MinVarPOTaskTester
from .mopo import MOPOTaskTester