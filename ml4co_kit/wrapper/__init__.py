r"""
Wrapper Module.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from .base import WrapperBase

# Routing Problems (TSP)
from .routing.tsp.atsp import ATSPWrapper
from .routing.tsp.op import OPWrapper
from .routing.tsp.pctsp import PCTSPWrapper
from .routing.tsp.spctsp import SPCTSPWrapper
from .routing.tsp.tsp import TSPWrapper

# Routing Problems (VRP)
from .routing.vrp.cvrp import CVRPWrapper
from .routing.vrp.cvrpb import CVRPBWrapper
from .routing.vrp.cvrpbl import CVRPBLWrapper
from .routing.vrp.cvrpbltw import CVRPBLTWWrapper
from .routing.vrp.cvrpbtw import CVRPBTWWrapper
from .routing.vrp.cvrpl import CVRPLWrapper
from .routing.vrp.cvrpltw import CVRPLTWWrapper
from .routing.vrp.cvrptw import CVRPTWWrapper

# Graph Problems
from .graph.mcl import MClWrapper
from .graph.mcut import MCutWrapper
from .graph.mis import MISWrapper
from .graph.mvc import MVCWrapper

# Portfolio Problems
from .portfolio.maxretpo import MaxRetPOWrapper
from .portfolio.minvarpo import MinVarPOWrapper
from .portfolio.mopo import MOPOWrapper

# QAP Problems
from .qap.gm import GMWrapper
from .qap.ged import GEDWrapper
from .qap.kqap import KQAPWrapper

# SAT Problems
from .sat.satp import SATPWrapper
from .sat.sata import SATAWrapper
