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
from .routing.atsp import ATSPGenTester
from .routing.cvrp import CVRPGenTester
from .routing.op import OPGenTester
from .routing.pctsp import PCTSPGenTester
from .routing.spctsp import SPCTSPGenTester
from .routing.tsp import TSPGenTester


# Graph Problems
from .graph.mcut import MCutGenTester
from .graph.mcl import MClGenTester
from .graph.mis import MISGenTester
from .graph.mvc import MVCGenTester


# Portfolio Problems
from .portfolio.minvarpo import MinVarPOGenTester
from .portfolio.maxretpo import MaxRetPOGenTester
from .portfolio.mopo import MOPOGenTester


# SAT Problems
from .sat.satp import SATPGenTester
from .sat.sata import SATAGenTester
from .sat.usatc import USATCGenTester