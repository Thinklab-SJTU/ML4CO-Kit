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
from .routing.atsp import ATSPTaskTester
from .routing.cvrp import CVRPTaskTester
from .routing.tsp import TSPTaskTester
from .routing.pctsp import PCTSPTaskTester

# Graph Problems
from .graph.mcl import MClTaskTester
from .graph.mcut import MCutTaskTester
from .graph.mis import MISTaskTester
from .graph.mvc import MVCTaskTester

# Graph Set Problems
from .graphset.gm import GMTaskTester
from .graphset.ged import GEDTaskTester

# Portfolio Problems
from .portfolio.maxretpo import MaxRetPOTaskTester
from .portfolio.minvarpo import MinVarPOTaskTester
from .portfolio.mopo import MOPOTaskTester