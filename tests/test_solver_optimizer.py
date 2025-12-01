r"""
Test Solver Module.
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


import os
import sys
from typing import Type
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)


# Checker
from ml4co_kit.utils.env_utils import EnvChecker
env_checker = EnvChecker()


# Get solvers to be tested (no torch used)
from tests.solver_optimizer_test import SolverTesterBase
from tests.solver_optimizer_test import (
    # solver testers
    ConcordeSolverTester,
    GAEAXSolverTester,
    GpDegreeSolverTester, 
    HGSSolverTester, 
    ILSSolverTester, 
    InsertionSolverTester, 
    ISCOSolverTester,
    KaMISSolverTester, 
    LcDegreeSolverTester,
    LKHSolverTester,
    ORSolverTester,
    RandomSolverTester,
    SCIPSolverTester,
    SMSolverTester,
    IPFPSolverTester,
    RRWMSolverTester,
    # optimizer testers
    CVRPLSOptimizerTester,
    ISCOOptimizerTester,
)

basic_tester_class_list = [
    # solver testers
    ConcordeSolverTester, 
    GAEAXSolverTester,
    GpDegreeSolverTester, 
    HGSSolverTester, 
    ILSSolverTester, 
    InsertionSolverTester, 
    ISCOSolverTester,
    LcDegreeSolverTester,
    LKHSolverTester,
    ORSolverTester,
    RandomSolverTester,
    SCIPSolverTester,
    SMSolverTester,
    IPFPSolverTester,
    RRWMSolverTester,
    # optimizer testers
    CVRPLSOptimizerTester,
    ISCOOptimizerTester,
]
if env_checker.system == "Linux":
    basic_tester_class_list.append(KaMISSolverTester)


# Gurobi
env_checker.gurobi_support = False # Currently, Github Actions does not support Gurobi
if env_checker.check_gurobi():
    from tests.solver_optimizer_test import GurobiSolverTester
    basic_tester_class_list.append(GurobiSolverTester)
   
    
# Get solvers to be tested (torch used)
if env_checker.check_torch():
    from tests.solver_optimizer_test import (
        RLSASolverTester, 
        NeuroLKHSolverTester,
        AStarSolverTester,
        NGMSolverTester,
        GennAStarSolverTester,
    )
    torch_tester_class_list = [
        RLSASolverTester,
        NeuroLKHSolverTester,
        AStarSolverTester,
        NGMSolverTester,
        GennAStarSolverTester,
    ]
if env_checker.check_gnn4co():
    from tests.solver_optimizer_test import (
        GNN4COBeamSolverTester, 
        GNN4COGreedySolverTester,
        GNN4COMCTSSolverTester,
        MCTSOptimizerTester,
        RLSAOptimizerTester,
        TwoOptOptimizerTester,
    )
    torch_tester_class_list += [
        GNN4COBeamSolverTester, 
        GNN4COGreedySolverTester,
        GNN4COMCTSSolverTester,
        MCTSOptimizerTester,
        RLSAOptimizerTester,
        TwoOptOptimizerTester,
    ]
    

# Test Solver
def test_solver_optimizer():
    # Basic Solvers
    for tester_class in basic_tester_class_list:
        tester_class: Type[SolverTesterBase]
        tester_class().test()
    
    # Torch Solvers
    for tester_class in torch_tester_class_list:
        tester_class: Type[SolverTesterBase]
        tester_class(device="cpu").test()
        if env_checker.check_cuda():
            tester_class(device="cuda").test()


# Main
if __name__ == "__main__":
    test_solver_optimizer()