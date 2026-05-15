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

# Routing Problems
from .routing.atsp import ATSPWrapper
from .routing.cvrp import CVRPWrapper
from .routing.op import OPWrapper
from .routing.pctsp import PCTSPWrapper
from .routing.spctsp import SPCTSPWrapper
from .routing.tsp import TSPWrapper

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