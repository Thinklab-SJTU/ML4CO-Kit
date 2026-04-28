r"""
MCMC Optimizer Tester.
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


from ml4co_kit import TASK_TYPE, NearestSolver, MCMCOptimizer, LcDegreeSolver
from tests.solver_optimizer_test.base import SolverTesterBase


# Optimizers
tsp_optimizer = MCMCOptimizer(tau_start=0.01, tau_end=0.001, num_steps=int(1e6))
cvrp_optimizer = MCMCOptimizer(tau_start=0.01, tau_end=0.001, num_steps=int(1e6))
mis_optimizer = MCMCOptimizer(tau_start=1.0, tau_end=0.001, num_steps=int(1e6))


class RoutingMCMCOptimizerTester(SolverTesterBase):
    def __init__(self):
        super(RoutingMCMCOptimizerTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=NearestSolver,
            test_task_type_list=[TASK_TYPE.TSP, TASK_TYPE.CVRP],
            test_args_list=[
                {"optimizer": tsp_optimizer}, # TSP
                {"optimizer": cvrp_optimizer}, # CVRP
            ],
            exclude_test_files_list=[[],[]]
        )
        
    def pre_test(self):
        pass


class GraphMCMCOptimizerTester(SolverTesterBase):
    def __init__(self):
        super(GraphMCMCOptimizerTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=LcDegreeSolver,
            test_task_type_list=[TASK_TYPE.MIS],
            test_args_list=[
                {"optimizer": mis_optimizer}, # MIS
            ],
            exclude_test_files_list=[[]]
        )
        
    def pre_test(self):
        pass