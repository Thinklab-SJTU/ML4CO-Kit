r"""
Generator for UNSAT-C (Unsat-core Variable Prediction) task instances.
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
from ml4co_kit.task.sat import USATCTask
from ml4co_kit.generator.sat.base import SATGeneratorBase, SAT_DISTRIBUTION


class USATCGenerator(SATGeneratorBase):
    """Generator for UNSAT-C (Unsat-core Variable Prediction) task instances."""
    
    def __init__(
        self,
        distribution_type: SAT_DISTRIBUTION = SAT_DISTRIBUTION.SR,
        precision: Union[np.float32, np.float64] = np.float32,
        vars_num: int = 50,
        clauses_num: Optional[int] = None,
        clause_length: int = 3,
        seed: Optional[int] = None,
    ):
        # Super Initialization
        super(USATCGenerator, self).__init__(
            task_type=TASK_TYPE.USATC,
            distribution_type=distribution_type,
            precision=precision,
            vars_num=vars_num,
            clauses_num=clauses_num,
            clause_length=clause_length,
            seed=seed
        )
        
        # Generation function dictionary
        # UNSAT-C only generates UNSAT instances
        self.generate_func_dict = {
            SAT_DISTRIBUTION.SR: self._generate_sr,
            SAT_DISTRIBUTION.UNIFORM_RANDOM: self._generate_uniform_random_unsat,
            SAT_DISTRIBUTION.K_CLIQUE: self._generate_k_clique,
        }
    
    def _create_instance(
        self, 
        clauses: List[List[int]], 
        unsat_core_vars: np.ndarray,
        **kwargs
    ) -> USATCTask:
        """Create a UNSAT-C task instance."""
        task = USATCTask(precision=self.precision)
        task.from_data(
            clauses=clauses,
            vars_num=self.vars_num
        )
        task.unsat_core_vars = unsat_core_vars
        return task
    
    def _generate_sr(self) -> USATCTask:
        """
        Generate UNSAT instance using SR method and extract core variables.
        """
        try:
            from pysat.solvers import Glucose3
            from pysat.examples.musx import MUSX
        except ImportError:
            print("Warning: python-sat not installed, using simple UNSAT generation")
            return self._generate_simple_unsat()
        
        max_attempts = 100
        for attempt in range(max_attempts):
            solver = Glucose3()
            clauses = []
            
            # Incrementally add clauses until UNSAT
            while len(clauses) < self.clauses_num * 2:
                k = self._sample_clause_length()
                clause = self._generate_random_clause(k)
                
                solver.add_clause(clause)
                
                if not solver.solve():
                    # Reached UNSAT
                    clauses.append(clause)
                    solver.delete()
                    
                    # Extract UNSAT core
                    unsat_core_vars = self._extract_unsat_core(clauses)
                    
                    if unsat_core_vars is not None:
                        return self._create_instance(clauses, unsat_core_vars)
                    break
                
                clauses.append(clause)
            
            solver.delete()
        
        # Fallback
        return self._generate_simple_unsat()
    
    def _sample_clause_length(self) -> int:
        """Sample clause length for SR generation."""
        p_k_2 = 0.3
        p_geo = 0.4
        
        if np.random.random() < p_k_2:
            k_base = 1
        else:
            k_base = 2
        
        k = k_base + np.random.geometric(p_geo)
        return min(k, self.vars_num)
    
    def _generate_uniform_random_unsat(self) -> USATCTask:
        """Generate uniform random UNSAT instance."""
        try:
            from pysat.solvers import Glucose3
        except ImportError:
            return self._generate_simple_unsat()
        
        max_attempts = 100
        for attempt in range(max_attempts):
            clauses = []
            
            # Generate many clauses to increase chance of UNSAT
            num_clauses = int(self.clauses_num * 1.5)
            
            for _ in range(num_clauses):
                clause = self._generate_random_clause()
                clauses.append(clause)
            
            # Check if UNSAT
            solver = Glucose3()
            for clause in clauses:
                solver.add_clause(clause)
            
            if not solver.solve():
                solver.delete()
                
                # Extract UNSAT core
                unsat_core_vars = self._extract_unsat_core(clauses)
                
                if unsat_core_vars is not None:
                    return self._create_instance(clauses, unsat_core_vars)
            
            solver.delete()
        
        # Fallback
        return self._generate_simple_unsat()
    
    def _generate_k_clique(self) -> USATCTask:
        """
        Generate UNSAT instance by reduction from k-clique problem.
        Following G4SATBench approach.
        """
        # This is a simplified version
        # Full implementation would encode k-clique into CNF
        return self._generate_sr()
    
    def _generate_simple_unsat(self) -> USATCTask:
        """
        Generate simple UNSAT instance with contradiction.
        Fallback method when pysat is not available.
        """
        clauses = []
        
        # Generate regular clauses
        for _ in range(self.clauses_num - 2):
            clause = self._generate_random_clause()
            clauses.append(clause)
        
        # Add contradiction: x1 and ¬x1
        clauses.append([1])
        clauses.append([-1])
        
        # Core variables: just variable 1
        unsat_core_vars = np.zeros(self.vars_num, dtype=bool)
        unsat_core_vars[0] = True
        
        return self._create_instance(clauses, unsat_core_vars)
    
    def _extract_unsat_core(self, clauses: List[List[int]]) -> Optional[np.ndarray]:
        """Extract UNSAT core variables using MUSX."""
        try:
            from pysat.examples.musx import MUSX
            
            # Use MUSX to find minimal unsatisfiable subset
            musx = MUSX(clauses)
            core_clause_indices = musx.compute()
            
            if not core_clause_indices:
                return None
            
            # Extract variables from core clauses
            core_vars = set()
            for idx in core_clause_indices:
                for lit in clauses[idx]:
                    core_vars.add(abs(lit))
            
            # Convert to boolean array
            unsat_core_vars = np.zeros(self.vars_num, dtype=bool)
            for var in core_vars:
                if var <= self.vars_num:
                    unsat_core_vars[var - 1] = True
            
            return unsat_core_vars
            
        except Exception as e:
            print(f"Warning: Failed to extract UNSAT core: {e}")
            # Return all variables as core (conservative estimate)
            return np.ones(self.vars_num, dtype=bool)
