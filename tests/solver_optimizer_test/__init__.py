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
from ml4co_kit.utils.env_utils import EnvChecker
env_checker = EnvChecker()
if env_checker.check_gnn4co():
    from .solver.beam import BeamSolverTester
    from .solver.greedy import GreedySolverTester
    from .solver.mcts import MCTSSolverTester
    from .optimizer.two_opt import TwoOptOptimizerTester
    from .optimizer.mcts import MCTSOptimizerTester
    from .optimizer.rlsa import RLSAOptimizerTester
if env_checker.check_torch():
    from .solver.neurolkh import NeuroLKHSolverTester
    from .solver.rlsa import RLSASolverTester


# Load other solver testers
from .base import SolverTesterBase
from .solver.concorde import ConcordeSolverTester
from .solver.ga_eax import GAEAXSolverTester
from .solver.gp_degree import GpDegreeSolverTester
from .solver.gurobi import GurobiSolverTester
from .solver.hgs import HGSSolverTester
from .solver.ils import ILSSolverTester
from .solver.insertion import InsertionSolverTester
from .solver.kamis import KaMISSolverTester
from .solver.lc_degree import LcDegreeSolverTester
from .solver.lkh import LKHSolverTester
from .solver.ortools import ORSolverTester
from .solver.scip import SCIPSolverTester