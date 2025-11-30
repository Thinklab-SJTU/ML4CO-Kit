r"""
Generator for SAT-P (Satisfiability Prediction) task instances.
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
from ml4co_kit.task.sat import SATPTask
from ml4co_kit.generator.sat.base import SATGeneratorBase, SAT_DISTRIBUTION


class SATPGenerator(SATGeneratorBase):
    """Generator for SAT-P (Satisfiability Prediction) task instances."""
    
    def __init__(
        self,
        distribution_type: SAT_DISTRIBUTION = SAT_DISTRIBUTION.UNIFORM_RANDOM,
        precision: Union[np.float32, np.float64] = np.float32,
        vars_num: int = 50,
        clauses_num: Optional[int] = None,
        clause_length: int = 3,
        seed: Optional[int] = None,
    ):
        # Super Initialization
        super(SATPGenerator, self).__init__(
            task_type=TASK_TYPE.SATP,
            distribution_type=distribution_type,
            precision=precision,
            vars_num=vars_num,
            clauses_num=clauses_num,
            clause_length=clause_length,
            seed=seed
        )
        
        # Generation function dictionary
        self.generate_func_dict = {
            SAT_DISTRIBUTION.UNIFORM_RANDOM: self._generate_uniform_random,
            SAT_DISTRIBUTION.PHASE_TRANSITION: self._generate_phase_transition,
            SAT_DISTRIBUTION.PLANTED: self._generate_planted,
            SAT_DISTRIBUTION.SR: self._generate_sr,
        }
    
    def _create_instance(
        self, 
        clauses: List[List[int]], 
        satisfiable: bool,
        **kwargs
    ) -> SATPTask:
        """Create a SAT-P task instance."""
        task = SATPTask(precision=self.precision)
        task.from_data(
            clauses=clauses,
            vars_num=self.vars_num,
            satisfiable=satisfiable
        )
        return task
    
    def _generate_uniform_random(self) -> SATPTask:
        """Generate standard k-SAT instance with uniform random clauses."""
        clauses = []
        for _ in range(self.clauses_num):
            clause = self._generate_random_clause()
            clauses.append(clause)
        
        # Use SAT solver to check satisfiability
        satisfiable = self._check_satisfiable(clauses)
        
        return self._create_instance(clauses, satisfiable)
    
    def _generate_phase_transition(self) -> SATPTask:
        """Generate instance near the satisfiability phase transition."""
        # Use precise phase transition formula
        if self.clause_length == 3:
            # G4SATBench formula: α_c ≈ 4.258 + 58.26 * n^(-2/3)
            alpha_c = 4.258 + 58.26 * (self.vars_num ** (-2.0/3.0))
            num_clauses = int(alpha_c * self.vars_num)
        else:
            alpha_c = (2 ** self.clause_length) * np.log(2)
            num_clauses = int(alpha_c * self.vars_num)
        
        clauses = []
        for _ in range(num_clauses):
            clause = self._generate_random_clause()
            clauses.append(clause)
        
        satisfiable = self._check_satisfiable(clauses)
        
        return self._create_instance(clauses, satisfiable)
    
    def _generate_planted(self) -> SATPTask:
        """Generate SAT instance with planted solution (guaranteed SAT)."""
        # Generate planted solution
        solution = np.random.randint(0, 2, self.vars_num).astype(bool)
        
        clauses = []
        for _ in range(self.clauses_num):
            # Generate clause that is satisfied by the planted solution
            clause = self._generate_satisfying_clause(solution)
            clauses.append(clause)
        
        return self._create_instance(clauses, satisfiable=True)
    
    def _generate_satisfying_clause(self, solution: np.ndarray) -> List[int]:
        """Generate a clause satisfied by the given solution."""
        while True:
            clause = self._generate_random_clause()
            
            # Check if clause is satisfied
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if (lit > 0 and solution[var_idx]) or (lit < 0 and not solution[var_idx]):
                    satisfied = True
                    break
            
            if satisfied:
                return clause
            
            # If not satisfied, flip one literal to make it satisfied
            lit_to_flip = clause[0]
            var_idx = abs(lit_to_flip) - 1
            if solution[var_idx]:
                clause[0] = abs(lit_to_flip)  # Make it positive
            else:
                clause[0] = -abs(lit_to_flip)  # Make it negative
            return clause
    
    def _generate_sr(self) -> SATPTask:
        """
        Generate instance using Satisfiability Resolution (SR) method.
        This creates instances near the SAT/UNSAT boundary.
        """
        try:
            from pysat.solvers import Glucose3
        except ImportError:
            print("Warning: python-sat not installed, falling back to uniform random")
            return self._generate_uniform_random()
        
        max_attempts = 100
        for attempt in range(max_attempts):
            solver = Glucose3()
            clauses = []
            
            # Incrementally add clauses until UNSAT or reaching max
            while len(clauses) < self.clauses_num * 2:  # Allow more clauses
                # Generate random clause with variable length
                k = self._sample_clause_length()
                clause = self._generate_random_clause(k)
                
                # Add to solver and check
                solver.add_clause(clause)
                
                if solver.solve():
                    clauses.append(clause)
                else:
                    # Reached UNSAT
                    solver.delete()
                    
                    # Return SAT instance (without last clause)
                    if np.random.random() < 0.5 and len(clauses) > 0:
                        return self._create_instance(clauses, satisfiable=True)
                    # Return UNSAT instance (with last clause)
                    else:
                        clauses.append(clause)
                        return self._create_instance(clauses, satisfiable=False)
            
            solver.delete()
            
            # If reached max clauses and still SAT
            if len(clauses) >= self.clauses_num:
                return self._create_instance(clauses[:self.clauses_num], satisfiable=True)
        
        # Fallback
        return self._generate_uniform_random()
    
    def _sample_clause_length(self) -> int:
        """Sample clause length for SR generation (following G4SATBench)."""
        p_k_2 = 0.3  # Probability of k=2
        p_geo = 0.4  # Geometric distribution parameter
        
        if np.random.random() < p_k_2:
            k_base = 1
        else:
            k_base = 2
        
        k = k_base + np.random.geometric(p_geo)
        return min(k, self.vars_num)
    
    def _check_satisfiable(self, clauses: List[List[int]]) -> bool:
        """Check if clauses are satisfiable using SAT solver."""
        try:
            from pysat.solvers import Glucose3
            solver = Glucose3()
            for clause in clauses:
                solver.add_clause(clause)
            result = solver.solve()
            solver.delete()
            return result
        except ImportError:
            # If pysat not available, randomly assign (for testing only)
            print("Warning: python-sat not installed, randomly assigning satisfiability")
            return np.random.random() < 0.5
