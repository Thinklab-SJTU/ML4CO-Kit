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


# Load env_checker
from ml4co_kit import env_checker
if env_checker.check_cp311_or_later():
    from .routing.vrp.cvrp_o import OVRPWrapperTester
    from .routing.vrp.cvrpb import CVRPBWrapperTester
    from .routing.vrp.cvrpb_o import OVRPBWrapperTester
    from .routing.vrp.cvrpbl import CVRPBLWrapperTester
    from .routing.vrp.cvrpbl_o import OVRPBLWrapperTester
    from .routing.vrp.cvrpbltw import CVRPBLTWWrapperTester
    from .routing.vrp.cvrpbltw_o import OVRPBLTWWrapperTester
    from .routing.vrp.cvrpbtw import CVRPBTWWrapperTester
    from .routing.vrp.cvrpbtw_o import OVRPBTWWrapperTester
    from .routing.vrp.cvrpl import CVRPLWrapperTester
    from .routing.vrp.cvrpl_o import OVRPLWrapperTester
    from .routing.vrp.cvrpltw import CVRPLTWWrapperTester
    from .routing.vrp.cvrpltw_o import OVRPLTWWrapperTester
    from .routing.vrp.cvrptw import CVRPTWWrapperTester
    from .routing.vrp.cvrptw_o import OVRPTWWrapperTester

# Base Class
from .base import WrapperTesterBase

# Routing Problems
from .routing.tsp.atsp import ATSPWrapperTester
from .routing.tsp.op import OPWrapperTester
from .routing.tsp.pctsp import PCTSPWrapperTester
from .routing.tsp.spctsp import SPCTSPWrapperTester
from .routing.tsp.tsp import TSPWrapperTester
from .routing.vrp.cvrp import CVRPWrapperTester

# Graph Problems
from .graph.mcl import MClWrapperTester
from .graph.mis import MISWrapperTester
from .graph.mvc import MVCWrapperTester
from .graph.mcut import MCutWrapperTester

# Portfolio Problems
from .portfolio.maxretpo import MaxRetPOWrapperTester
from .portfolio.minvarpo import MinVarPOWrapperTester
from .portfolio.mopo import MOPOWrapperTester

# SAT Problems
from .sat.satp import SATPWrapperTester
from .sat.sata import SATAWrapperTester