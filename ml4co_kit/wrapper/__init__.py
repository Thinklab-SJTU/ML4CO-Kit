r"""
Wrapper Module.
"""

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

# SAT Problems
from .sat.satp import SATPWrapper
from .sat.sata import SATAWrapper
from .sat.usatc import USATCWrapper