r"""
Solver Module.
"""

from .base import SolverBase, SOLVER_TYPE
from .greedy import GreedySolver
from .lkh import LKHSolver
from .concorde import ConcordeSolver
from .kamis import KaMISSolver
from .gp_degree import GpDegreeSolver
from .lc_degree import LcDegreeSolver
from .insertion import InsertionSolver
from .mcts import MCTSSolver
from .rlsa import RLSASolver
from .hgs import HGSSolver