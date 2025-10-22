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
if env_checker.check_gnn4co():
    from .beam import BeamSolver
    from .greedy import GreedySolver
    from .mcts import MCTSSolver
if env_checker.check_torch():
    from .neurolkh import NeuroLKHSolver
    from .rlsa import RLSASolver


# Load other solvers
from .base import SolverBase, SOLVER_TYPE
from .concorde import ConcordeSolver
from .ga_eax import GAEAXSolver
from .gp_degree import GpDegreeSolver
from .gurobi import GurobiSolver
from .hgs import HGSSolver
from .ils import ILSSolver
from .insertion import InsertionSolver
from .kamis import KaMISSolver
from .lc_degree import LcDegreeSolver
from .lkh import LKHSolver
from .ortools import ORSolver
from .scip import SCIPSolver