r"""
Generator Test Module.
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
from .base import GenTesterBase

# Routing Problems
from .atsp import ATSPGenTester
from .cvrp import CVRPGenTester
from .op import OPGenTester
from .pctsp import PCTSPGenTester
from .spctsp import SPCTSPGenTester
from .tsp import TSPGenTester

# Graph Problems
from .mcut import MCutGenTester
from .mcl import MClGenTester
from .mis import MISGenTester
from .mvc import MVCGenTester

# Portfolio Problems
from .minvarpo import MinVarPOGenTester
from .maxretpo import MaxRetPOGenTester
from .mopo import MOPOGenTester
