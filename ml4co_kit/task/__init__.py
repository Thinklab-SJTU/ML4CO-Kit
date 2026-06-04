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

# Routing Task (Base)
from .routing.base import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE

# Routing Task (TSP)
from .routing.tsp.atsp import ATSPTask
from .routing.tsp.op import OPTask
from .routing.tsp.pctsp import PCTSPTask
from .routing.tsp.spctsp import SPCTSPTask
from .routing.tsp.tsp import TSPTask

# Routing Task (VRP)
from .routing.vrp.cvrp import CVRPTask
from .routing.vrp.cvrpb import CVRPBTask
from .routing.vrp.cvrpbl import CVRPBLTask
from .routing.vrp.cvrpbltw import CVRPBLTWTask
from .routing.vrp.cvrpbtw import CVRPBTWTask
from .routing.vrp.cvrpl import CVRPLTask
from .routing.vrp.cvrpltw import CVRPLTWTask
from .routing.vrp.cvrptw import CVRPTWTask

# SAT Task
from .sat.base import SATTaskBase
from .sat.satp import SATPTask
from .sat.sata import SATATask

# MILP / LP Task
from .milp.base import MILPTaskBase
from .milp.lp import LPTask
from .milp.milp import MILPTask