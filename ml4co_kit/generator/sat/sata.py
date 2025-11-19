r"""
Generator for SAT-A (Satisfying Assignment Prediction) task instances.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import Union, List, Optional
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.sat import SATATask
from ml4co_kit.generator.sat.base import SATGeneratorBase, SAT_DISTRIBUTION


class SATAGenerator(SATGeneratorBase):
    """Generator for SAT-A (Satisfying Assignment Prediction) task instances."""
    
    def __init__(
        self,
        distribution_type: SAT_DISTRIBUTION = SAT_DISTRIBUTION.PLANTED,
        precision: Union[np.float32, np.float64] = np.float32,
        vars_num: int = 50,
        clauses_num: Optional[int] = None,
        clause_length: int = 3,
        seed: Optional[int] = None,
    ):
        # Super Initialization
        super(SATAGenerator, self).__init__(
            task_type=TASK_TYPE.SATA,
            distribution_type=distribution_type,
            precision=precision,
            vars_num=vars_num,
            clauses_num=clauses_num,
            clause_length=clause_length,
            seed=seed
        )
        
        # Generation function dictionary
        # SAT-A only generates SAT instances
        self.generate_func_dict = {
            SAT_DISTRIBUTION.PLANTED: self._generate_planted,
            SAT_DISTRIBUTION.UNIFORM_RANDOM: self._generate_uniform_random_sat,
        }
    
    def _create_instance(
        self, 
        clauses: List[List[int]], 
        solution: np.ndarray,
        **kwargs
    ) -> SATATask:
        """Create a SAT-A task instance."""
        task = SATATask(precision=self.precision)
        task.from_data(
            clauses=clauses,
            vars_num=self.vars_num,
            sol=solution,
            ref=True  # This is a reference solution
        )
        return task
    
    def _generate_planted(self) -> SATATask:
        """Generate SAT instance with planted solution."""
        # Generate planted solution
        solution = np.random.randint(0, 2, self.vars_num).astype(bool)
        
        clauses = []
        for _ in range(self.clauses_num):
            # Generate clause satisfied by the solution
            clause = self._generate_satisfying_clause(solution)
            clauses.append(clause)
        
        return self._create_instance(clauses, solution)
    
    def _generate_satisfying_clause(self, solution: np.ndarray) -> List[int]:
        """Generate a clause satisfied by the given solution."""
        while True:
            clause = self._generate_random_clause()
            
            # Check if at least one literal is satisfied
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if (lit > 0 and solution[var_idx]) or (lit < 0 and not solution[var_idx]):
                    satisfied = True
                    break
            
            if satisfied:
                return clause
            
            # If not satisfied, flip one literal to satisfy it
            lit_to_flip = np.random.choice(len(clause))
            var_idx = abs(clause[lit_to_flip]) - 1
            
            if solution[var_idx]:
                clause[lit_to_flip] = abs(clause[lit_to_flip])  # Make positive
            else:
                clause[lit_to_flip] = -abs(clause[lit_to_flip])  # Make negative
            
            return clause
    
    def _generate_uniform_random_sat(self) -> SATATask:
        """
        Generate uniform random SAT instance and find solution using solver.
        Warning: This may fail if instance is UNSAT.
        """
        try:
            from pysat.solvers import Glucose3
        except ImportError:
            print("Warning: python-sat not installed, using planted generation")
            return self._generate_planted()
        
        max_attempts = 100
        for attempt in range(max_attempts):
            clauses = []
            for _ in range(self.clauses_num):
                clause = self._generate_random_clause()
                clauses.append(clause)
            
            # Check if SAT and get solution
            solver = Glucose3()
            for clause in clauses:
                solver.add_clause(clause)
            
            if solver.solve():
                # Get model (solution)
                model = solver.get_model()
                solver.delete()
                
                # Convert model to boolean array
                solution = np.zeros(self.vars_num, dtype=bool)
                for lit in model:
                    if abs(lit) <= self.vars_num:
                        var_idx = abs(lit) - 1
                        solution[var_idx] = (lit > 0)
                
                return self._create_instance(clauses, solution)
            
            solver.delete()
        
        # If all attempts failed, use planted generation
        print(f"Warning: Failed to generate SAT instance after {max_attempts} attempts, using planted")
        return self._generate_planted()
