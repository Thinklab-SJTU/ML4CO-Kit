r"""
Solver Module.
"""

from .base import SolverBase, SOLVER_TYPE
from .beam import BeamSolver
from .concorde import ConcordeSolver
from .ga_eax import GAEAXSolver
from .gp_degree import GpDegreeSolver
from .greedy import GreedySolver
from .hgs import HGSSolver
from .insertion import InsertionSolver
from .kamis import KaMISSolver
from .lc_degree import LcDegreeSolver
from .lkh import LKHSolver
from .mcts import MCTSSolver
from .rlsa import RLSASolver