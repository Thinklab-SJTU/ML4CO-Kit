r"""
Wrapper Module.
"""

from .base import WrapperBase

# Routing Problems
from .atsp import ATSPWrapper
from .cvrp import CVRPWrapper
from .op import OPWrapper
from .pctsp import PCTSPWrapper
from .spctsp import SPCTSPWrapper
from .tsp import TSPWrapper

# Graph Problems
from .mcl import MClWrapper
from .mcut import MCutWrapper
from .mis import MISWrapper
from .mvc import MVCWrapper

# Portfolio Optimization Problems
from .maxretpo import MaxRetPOWrapper
from .minvarpo import MinVarPOWrapper
from .mopo import MOPOWrapper