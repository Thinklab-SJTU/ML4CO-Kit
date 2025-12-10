r"""
Wrapper Test Module.
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
from .base import WrapperTesterBase

# Routing Problems
from .routing.atsp import ATSPWrapperTester
from .routing.cvrp import CVRPWrapperTester
from .routing.op import OPWrapperTester
from .routing.pctsp import PCTSPWrapperTester
from .routing.spctsp import SPCTSPWrapperTester
from .routing.tsp import TSPWrapperTester

# Graph Problems
from .graph.mcl import MClWrapperTester
from .graph.mis import MISWrapperTester
from .graph.mvc import MVCWrapperTester
from .graph.mcut import MCutWrapperTester

# Graph Set Problem
from .graphset.gm import GMWrapperTester
from .graphset.ged import GEDWrapperTester

# Portfolio Problems
from .portfolio.maxretpo import MaxRetPOWrapperTester
from .portfolio.minvarpo import MinVarPOWrapperTester
from .portfolio.mopo import MOPOWrapperTester
