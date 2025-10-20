r"""
Generator Module.
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

# Base Generator
from .base import GeneratorBase

# Graph Generator
from .graph.base import (
    GraphGeneratorBase, GRAPH_TYPE, GRAPH_WEIGHT_TYPE, GraphWeightGenerator
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

# Routing Generator
from .routing.base import RoutingGeneratorBase
from .routing.atsp import ATSPGenerator, ATSP_TYPE
from .routing.cvrp import CVRPGenerator, CVRP_TYPE
from .routing.op import OPGenerator, OP_TYPE
from .routing.pctsp import PCTSPGenerator, PCTSP_TYPE
from .routing.spctsp import SPCTSPGenerator, SPCTSP_TYPE
from .routing.tsp import TSPGenerator, TSP_TYPE
