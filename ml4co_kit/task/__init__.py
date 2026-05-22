r"""
Task Module.
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


# Base Task
from .base import TaskBase, TASK_TYPE

# EDA Task
from .eda.base import EDA_BENCH
from .eda.edap import EDAPTask

# Graph Task
from .graph.base import GraphTaskBase
from .graph.mcl import MClTask
from .graph.mcut import MCutTask
from .graph.mis import MISTask
from .graph.mvc import MVCTask

# Portfolio Task
from .portfolio.base import PortfolioTaskBase
from .portfolio.minvarpo import MinVarPOTask
from .portfolio.maxretpo import MaxRetPOTask
from .portfolio.mopo import MOPOTask

# QAP Task
from .qap.base import QAPTaskBase, QAPGraphBase
from .qap.gm import GMTask, GMAffinityMatrixBuilder
from .qap.ged import GEDTask
from .qap.kqap import KQAPTask

# Routing Task
from .routing.base import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE
from .routing.atsp import ATSPTask
from .routing.cvrp import CVRPTask
from .routing.ovrp import OVRPTask
from .routing.op import OPTask
from .routing.pctsp import PCTSPTask
from .routing.spctsp import SPCTSPTask
from .routing.tsp import TSPTask
from .routing.vrpl import VRPLTask
from .routing.vrptw import VRPTWTask

# SAT Task
from .sat.base import SATTaskBase
from .sat.satp import SATPTask
from .sat.sata import SATATask

# 
