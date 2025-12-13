r"""
PySAT Solver Tester.
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


from ml4co_kit import PySATSolver
from ml4co_kit import SATPGenerator, SATAGenerator, USATCGenerator, SAT_TYPE


class PySATSolverTester:
    def __init__(self):
        self.solver = PySATSolver()
        self.sat_types = [
            SAT_TYPE.PHASE,
            SAT_TYPE.SR,
            SAT_TYPE.CA,
            SAT_TYPE.PS,
            SAT_TYPE.K_CLIQUE,
            SAT_TYPE.K_DOMSET,
            SAT_TYPE.K_VERCOV,
        ]
        
    def pre_test(self):
        pass

    def test(self):
        self.pre_test()
        print(f"\n--------------------------------------------------")
        print(f"Testing PySATSolver")
        
        self._test_satp()
        self._test_sata()
        self._test_usatc()
        
    def _test_satp(self):
        print("\nTest SAT-P tasks:")
        for sat_type in self.sat_types:
            generator = SATPGenerator(distribution_type=sat_type)
            task = generator.generate()
            self.solver.solve(task)
            print(f"  {sat_type.value}: Clauses={task.clauses_num}, Vars={task.vars_num}, Sol={task.sol}")

    def _test_sata(self):
        print("\nTest SAT-A tasks:")
        for sat_type in self.sat_types:
            generator = SATAGenerator(distribution_type=sat_type)
            task = generator.generate()
            self.solver.solve(task)
            eval_result = task.evaluate(task.sol)
            print(f"  {sat_type.value}: Clauses={task.clauses_num}, Vars={task.vars_num}, Satisfied={eval_result}")

    def _test_usatc(self):
        print("\nTest UNSAT-C tasks:")
        for sat_type in self.sat_types:
            generator = USATCGenerator(distribution_type=sat_type)
            task = generator.generate()
            self.solver.solve(task)
            core_vars = task.sol.sum()
            print(f"  {sat_type.value}: Clauses={task.clauses_num}, Vars={task.vars_num}, CoreVars={core_vars}")
