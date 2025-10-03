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
import importlib.util
from typing import Type
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)


# Check if torch is supported
found_torch = importlib.util.find_spec("torch")
if found_torch is not None:
    import torch
    TORCH_SUPPORT = True
    CUDA_SUPPORT = torch.cuda.is_available()
else:
    TORCH_SUPPORT = False
    CUDA_SUPPORT = False


# Check if gurobi is supported
import gurobipy as gp
try:
    env = gp.Env(empty=True)
    env.start()
    GUROBI_SUPPORT = True
except gp.GurobiError as e:
    GUROBI_SUPPORT = False


# Get solvers to be tested (no torch used)
from tests.solver_test import SolverTesterBase
from tests.solver_test import (
    ConcordeSolverTester, GAEAXSolverTester, GpDegreeSolverTester, 
    HGSSolverTester, InsertionSolverTester, KaMISSolverTester,
    LcDegreeSolverTester, LKHSolverTester
)
basic_solver_class_list = [
    ConcordeSolverTester, GAEAXSolverTester, GpDegreeSolverTester, 
    HGSSolverTester, InsertionSolverTester, KaMISSolverTester,
    LcDegreeSolverTester, LKHSolverTester
]


# Gurobi
if GUROBI_SUPPORT:
    from tests.solver_test import GurobiSolverTester
    basic_solver_class_list.append(GurobiSolverTester)
   
    
# Get solvers to be tested (torch used)
if TORCH_SUPPORT:
    from tests.solver_test import (
        GreedySolverTester, RLSASolverTester, MCTSSolverTester
    )
    torch_solver_class_list = [
        GreedySolverTester, RLSASolverTester, MCTSSolverTester
    ]


if __name__ == "__main__":
    # Basic Solvers
    for solver_class in basic_solver_class_list:
        solver_class: Type[SolverTesterBase]
        solver_class().test()
    
    # Torch Solvers
    for solver_class in torch_solver_class_list:
        solver_class: Type[SolverTesterBase]
        solver_class(device="cpu").test()
        if CUDA_SUPPORT:
            solver_class(device="cuda").test()