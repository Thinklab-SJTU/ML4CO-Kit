r"""
Solver Test Module.
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


# Check if torch is supported
import importlib.util
found_torch = importlib.util.find_spec("torch")

if found_torch is not None:
    from .beam import BeamSolverTester
    from .greedy import GreedySolverTester
    from .mcts import MCTSSolverTester
    from .neurolkh import NeuroLKHSolverTester
    from .rlsa import RLSASolverTester
else:
    TORCH_SUPPORT = False


# Load other solver testers
from .base import SolverTesterBase
from .concorde import ConcordeSolverTester
from .ga_eax import GAEAXSolverTester
from .gp_degree import GpDegreeSolverTester
from .gurobi import GurobiSolverTester
from .hgs import HGSSolverTester
from .ils import ILSSolverTester
from .insertion import InsertionSolverTester
from .kamis import KaMISSolverTester
from .lc_degree import LcDegreeSolverTester
from .lkh import LKHSolverTester
from .ortools import ORSolverTester
from .sm import SMSolverTester
from .rrwm import RRWMSolverTester
from .ipfp import IPFPSolverTester
from .astar import AStarSolverTester
from .genn_astar import GENN_AStarSolverTester
from .ngm import NGMSolverTester
