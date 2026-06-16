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
from .routing.tsp.atsp import ATSPGenTester
from .routing.tsp.op import OPGenTester
from .routing.tsp.pctsp import PCTSPGenTester
from .routing.tsp.spctsp import SPCTSPGenTester
from .routing.tsp.tsp import TSPGenTester
from .routing.vrp.cvrp import CVRPGenTester
from .routing.vrp.cvrpb import CVRPBGenTester
from .routing.vrp.cvrpbl import CVRPBLGenTester
from .routing.vrp.cvrpbltw import CVRPBLTWGenTester
from .routing.vrp.cvrpbtw import CVRPBTWGenTester
from .routing.vrp.cvrpl import CVRPLGenTester
from .routing.vrp.cvrpltw import CVRPLTWGenTester
from .routing.vrp.cvrptw import CVRPTWGenTester
from .routing.vrp.mtvrp import MTVRPGenTester

# Graph Problems
from .graph.mcut import MCutGenTester
from .graph.mcl import MClGenTester
from .graph.mis import MISGenTester
from .graph.mvc import MVCGenTester

# Portfolio Problems
from .portfolio.minvarpo import MinVarPOGenTester
from .portfolio.maxretpo import MaxRetPOGenTester
from .portfolio.mopo import MOPOGenTester

# QAP Problems
from .qap.gm import GMGenTester

# SAT Problems
from .sat.satp import SATPGenTester
from .sat.sata import SATAGenTester