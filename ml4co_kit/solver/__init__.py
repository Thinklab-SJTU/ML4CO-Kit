r"""
Solver Module.
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
from ml4co_kit.utils.env_utils import EnvChecker
env_checker = EnvChecker()
if env_checker.check_dreamplace():
    from .eda.dreamplace import DreamPlaceSolver
if env_checker.check_gnn4co():
    from .common.gnn4co import GNN4COSolver
if env_checker.check_torch():
    from .graph.fem import FEMSolver
    from .graph.rlsa import RLSASolver
    from .qap.pygm import PyGMSolver, PyGMToolsQAPSolver
    from .routing.neurolkh import NeuroLKHSolver

# Solver (Python Version)
if env_checker.check_cp311_or_later():
    from .routing.pyvrp import PyVRPSolver

# Basic Class
from .base import SolverBase, SOLVER_TYPE

# Common Solvers
from .common.gurobi import GurobiSolver
from .common.ils import ILSSolver
from .common.insertion import InsertionSolver
from .common.null import NullSolver
from .common.ortools import ORSolver
from .common.random import RandomSolver
from .common.scip import SCIPSolver

# Solvers for Graph Tasks
from .graph.gp_degree import GpDegreeSolver
from .graph.isco import ISCOSolver
from .graph.kamis import KaMISSolver
from .graph.lc_degree import LcDegreeSolver

# Solvers for Routing Tasks
from .routing.concorde import ConcordeSolver
from .routing.ga_eax import GAEAXSolver
from .routing.hgs import HGSSolver
from .routing.lkh import LKHSolver
from .routing.nearest import NearestSolver

# Solvers for SAT Tasks
from .sat.pysat import PySATSolver
