r"""
Generator Module.
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


# Base Generator
from .base import GeneratorBase

# Graph Generator
from .graph.base import (
    GraphGeneratorBase, GRAPH_TYPE, 
    GRAPH_WEIGHT_TYPE, GraphWeightGenerator
)
from .graph.mcl import MClGenerator
from .graph.mcut import MCutGenerator
from .graph.mis import MISGenerator
from .graph.mvc import MVCGenerator

# Portfolio Generator
from .portfolio.base import (
    PortfolioGeneratorBase, PO_TYPE, PortfolioDistributionArgs
)
from .portfolio.minvarpo import MinVarPOGenerator
from .portfolio.maxretpo import MaxRetPOGenerator
from .portfolio.mopo import MOPOGenerator

# Routing Generator (Base)
from .routing.base import RoutingGeneratorBase

# Routing Generator (TSP)
from .routing.tsp.atsp import ATSPGenerator, ATSP_TYPE
from .routing.tsp.op import OPGenerator, OP_TYPE
from .routing.tsp.pctsp import PCTSPGenerator, PCTSP_TYPE
from .routing.tsp.spctsp import SPCTSPGenerator, SPCTSP_TYPE
from .routing.tsp.tsp import TSPGenerator, TSP_TYPE

# Routing Generator (VRP)
from .routing.vrp.cvrp import CVRPGenerator, CVRP_TYPE
from .routing.vrp.cvrpb import CVRPBGenerator
from .routing.vrp.cvrpbl import CVRPBLGenerator
from .routing.vrp.cvrpbltw import CVRPBLTWGenerator
from .routing.vrp.cvrpbtw import CVRPBTWGenerator
from .routing.vrp.cvrpl import CVRPLGenerator
from .routing.vrp.cvrpltw import CVRPLTWGenerator
from .routing.vrp.cvrptw import CVRPTWGenerator
from .routing.vrp.mtvrp import MTVRPGenerator

# QAP Generator
from .qap.base import QAPGraphGenerator
from .qap.gm import GMGenerator, GM_TYPE

# SAT Generator
from .sat.base import SATGeneratorBase, SAT_TYPE
from .sat.satp import SATPGenerator
from .sat.sata import SATAGenerator
