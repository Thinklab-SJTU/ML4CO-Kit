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
from .atsp import ATSPWrapperTester
from .cvrp import CVRPWrapperTester
from .op import OPWrapperTester
from .pctsp import PCTSPWrapperTester
from .spctsp import SPCTSPWrapperTester
from .tsp import TSPWrapperTester

# Graph Problems
from .mcl import MClWrapperTester
from .mis import MISWrapperTester
from .mvc import MVCWrapperTester
from .mcut import MCutWrapperTester

# Portfolio Problems
from .maxretpo import MaxRetPOWrapperTester
from .minvarpo import MinVarPOWrapperTester
from .mopo import MOPOWrapperTester