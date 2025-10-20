r"""
Task Module.
"""

# Base Task
from .base import TaskBase, TASK_TYPE

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

# Routing Task
from .routing.base import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE
from .routing.atsp import ATSPTask
from .routing.cvrp import CVRPTask
from .routing.op import OPTask
from .routing.pctsp import PCTSPTask
from .routing.spctsp import SPCTSPTask
from .routing.tsp import TSPTask