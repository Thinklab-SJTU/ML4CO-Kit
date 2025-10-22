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
    ConcordeSolverTester,
    GAEAXSolverTester,
    GpDegreeSolverTester, 
    HGSSolverTester, 
    ILSSolverTester, 
    InsertionSolverTester, 
    KaMISSolverTester, 
    LcDegreeSolverTester,
    LKHSolverTester,
    ORSolverTester,
    SCIPSolverTester
)
basic_solver_class_list = [
    ConcordeSolverTester, 
    GAEAXSolverTester,
    GpDegreeSolverTester, 
    HGSSolverTester, 
    ILSSolverTester, 
    InsertionSolverTester, 
    LcDegreeSolverTester,
    LKHSolverTester,
    ORSolverTester,
    SCIPSolverTester
]
if env_checker.system == "Linux":
    basic_solver_class_list.append(KaMISSolverTester)


# Gurobi
env_checker.gurobi_support = False # Currently, Github Actions does not support Gurobi
if env_checker.check_gurobi():
    from tests.solver_optimizer_test import GurobiSolverTester
    basic_solver_class_list.append(GurobiSolverTester)
   
    
# Get solvers to be tested (torch used)
if env_checker.check_torch():
    from tests.solver_optimizer_test import (
        RLSASolverTester, 
        NeuroLKHSolverTester
    )
    torch_solver_class_list = [
        RLSASolverTester,
        NeuroLKHSolverTester,
    ]
if env_checker.check_gnn4co():
    from tests.solver_optimizer_test import (
        BeamSolverTester, 
        GreedySolverTester,
        MCTSSolverTester,
        MCTSOptimizerTester,
        RLSAOptimizerTester,
        TwoOptOptimizerTester,
    )
    torch_solver_class_list += [
        BeamSolverTester, 
        GreedySolverTester,
        MCTSSolverTester,
        MCTSOptimizerTester,
        RLSAOptimizerTester,
        TwoOptOptimizerTester
    ]
    

# Test Solver
def test_solver():
    # Basic Solvers
    for solver_class in basic_solver_class_list:
        solver_class: Type[SolverTesterBase]
        solver_class().test()
    
    # Torch Solvers
    for solver_class in torch_solver_class_list:
        solver_class: Type[SolverTesterBase]
        solver_class(device="cpu").test()
        if env_checker.check_cuda():
            solver_class(device="cuda").test()


# Main
if __name__ == "__main__":
    test_solver()