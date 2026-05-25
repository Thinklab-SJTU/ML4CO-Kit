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
    from .solver.common.gnn4co_beam import GNN4COBeamSolverTester
    from .solver.common.gnn4co_greedy import GNN4COGreedySolverTester
    from .solver.common.gnn4co_mcts import GNN4COMCTSSolverTester
    from .optimizer.two_opt import TwoOptOptimizerTester
    from .optimizer.mcts import MCTSOptimizerTester
if env_checker.check_torch():
    from .solver.graph.rlsa import RLSASolverTester
    from .solver.graph.fem import FEMSolverTester
    from .solver.routing.neurolkh import NeuroLKHSolverTester

# Check Python Version
if env_checker.check_cp311_or_later():
    from .solver.routing.pyvrp import PyVRPSolverTester

# Load other solver testers
from .base import SolverTesterBase
from .solver.common.gurobi import GurobiSolverTester
from .solver.common.ils import ILSSolverTester
from .solver.common.insertion import InsertionSolverTester
from .solver.common.ortools import ORSolverTester
from .solver.common.random import RandomSolverTester
from .solver.common.scip import SCIPSolverTester
from .solver.graph.gp_degree import GpDegreeSolverTester
from .solver.graph.isco import ISCOSolverTester
from .solver.graph.kamis import KaMISSolverTester
from .solver.graph.lc_degree import LcDegreeSolverTester
from .solver.qap.pygm import PyGMSolverTester
from .solver.routing.concorde import ConcordeSolverTester
from .solver.routing.ga_eax import GAEAXSolverTester
from .solver.routing.hgs import HGSSolverTester
from .solver.routing.lkh import LKHSolverTester
from .solver.routing.nearest import NearestSolverTester
from .solver.sat.pysat import PySATSolverTester

# Load other optimizer testers
from .optimizer.cvrp_ls import CVRPLSOptimizerTester
from .optimizer.fast_2opt import FastTwoOptOptimizerTester
from .optimizer.isco import ISCOOptimizerTester
from .optimizer.mcmc import RoutingMCMCOptimizerTester, GraphMCMCOptimizerTester