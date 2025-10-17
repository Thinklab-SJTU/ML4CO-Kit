r"""
Generator for Boolean Satisfiability Problem (SAT) instances.
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


import numpy as np
from typing import Union, List, Optional
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.logic.sat import SATTask
from ml4co_kit.generator.logic.base import LogicGeneratorBase, LOGIC_TYPE


class SATGenerator(LogicGeneratorBase):
    """Generator for SAT problem instances."""
    
    def __init__(
        self,
        distribution_type: LOGIC_TYPE = LOGIC_TYPE.UNIFORM_RANDOM,
        precision: Union[np.float32, np.float64] = np.float32,
        num_vars: int = 10,
        num_clauses: Optional[int] = None,
        clause_length: int = 3,
        seed: Optional[int] = None
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.SAT,
            distribution_type=distribution_type,
            precision=precision
        )
        
        # Set parameters
        self.set_parameters(num_vars, num_clauses, clause_length, seed)
        
        # Generation function dictionary
        self.generate_func_dict = {
            LOGIC_TYPE.RANDOM: self._generate_random,
            LOGIC_TYPE.UNIFORM_RANDOM: self._generate_uniform_random,
            LOGIC_TYPE.PLANTED: self._generate_planted,
            LOGIC_TYPE.PHASE_TRANSITION: self._generate_phase_transition,
            LOGIC_TYPE.INDUSTRIAL: self._generate_industrial,
        }
    
    def generate(
        self,
        num_vars: Optional[int] = None,
        num_clauses: Optional[int] = None,
        clause_length: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> SATTask:
        """Generate a SAT instance."""
        # Update parameters if provided
        if num_vars is not None:
            self.num_vars = num_vars
        if num_clauses is not None:
            self.num_clauses = num_clauses
        if clause_length is not None:
            self.clause_length = clause_length
        if seed is not None:
            self.seed = seed
            
        # Auto-calculate num_clauses if not provided
        if self.num_clauses is None:
            self.set_parameters(self.num_vars, None, self.clause_length, self.seed)
        
        # Set random seed
        self._set_random_seed()
        
        # Generate using the appropriate method
        generate_func = self.generate_func_dict[self.distribution_type]
        return generate_func(**kwargs)
    
    def _generate_random(self, **kwargs) -> SATTask:
        """Generate completely random SAT instance."""
        clauses = []
        
        for _ in range(self.num_clauses):
            # Random clause length between 1 and max(3, clause_length)
            clause_len = np.random.randint(1, max(3, self.clause_length) + 1)
            
            # Random variables for this clause
            vars_in_clause = np.random.choice(
                range(1, self.num_vars + 1), 
                size=clause_len, 
                replace=False
            )
            
            # Random polarities
            clause = []
            for var in vars_in_clause:
                if np.random.random() < 0.5:
                    clause.append(-var)  # Negative literal
                else:
                    clause.append(var)   # Positive literal
            
            clauses.append(clause)
        
        # Create SAT task
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        return sat_task
    
    def _generate_uniform_random(self, **kwargs) -> SATTask:
        """Generate uniform random k-SAT instance."""
        clauses = []
        
        for _ in range(self.num_clauses):
            # Fixed clause length (k-SAT)
            vars_in_clause = np.random.choice(
                range(1, self.num_vars + 1),
                size=self.clause_length,
                replace=False
            )
            
            # Random polarities
            clause = []
            for var in vars_in_clause:
                if np.random.random() < 0.5:
                    clause.append(-var)
                else:
                    clause.append(var)
            
            clauses.append(clause)
        
        # Create SAT task
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        return sat_task
    
    def _generate_planted(self, **kwargs) -> SATTask:
        """Generate SAT instance with planted (known) solution."""
        # Generate a random satisfying assignment
        planted_solution = np.random.choice([0, 1], size=self.num_vars)
        
        clauses = []
        max_attempts = self.num_clauses * 10  # Prevent infinite loops
        attempts = 0
        
        while len(clauses) < self.num_clauses and attempts < max_attempts:
            attempts += 1
            
            # Generate a random clause
            vars_in_clause = np.random.choice(
                range(1, self.num_vars + 1),
                size=self.clause_length,
                replace=False
            )
            
            clause = []
            for var in vars_in_clause:
                var_index = var - 1  # Convert to 0-based
                if np.random.random() < 0.5:
                    clause.append(-var)
                else:
                    clause.append(var)
            
            # Check if this clause is satisfied by the planted solution
            clause_satisfied = False
            for literal in clause:
                var_index = abs(literal) - 1
                if literal > 0 and planted_solution[var_index]:
                    clause_satisfied = True
                    break
                elif literal < 0 and not planted_solution[var_index]:
                    clause_satisfied = True
                    break
            
            # Only add clauses that are satisfied by the planted solution
            if clause_satisfied:
                clauses.append(clause)
        
        # Create SAT task with reference solution
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        sat_task.set_ref_assignment(planted_solution)
        return sat_task
    
    def _generate_phase_transition(self, **kwargs) -> SATTask:
        """Generate SAT instance near the satisfiability phase transition."""
        # Calculate phase transition ratio
        if self.clause_length == 3:
            phase_ratio = 4.26  # Known 3-SAT phase transition
        else:
            phase_ratio = 2**self.clause_length * np.log(2)
        
        # Add some randomness around the phase transition
        noise = kwargs.get('noise', 0.1)
        ratio = phase_ratio * (1 + np.random.uniform(-noise, noise))
        
        # Calculate number of clauses for this ratio
        num_clauses = int(ratio * self.num_vars)
        
        # Generate uniform random k-SAT with this ratio
        old_num_clauses = self.num_clauses
        self.num_clauses = num_clauses
        result = self._generate_uniform_random()
        self.num_clauses = old_num_clauses  # Restore original
        
        return result
    
    def _generate_industrial(self, **kwargs) -> SATTask:
        """Generate industrial-like SAT instance with structure."""
        # Industrial instances often have variable clause lengths and community structure
        clauses = []
        
        # Create some "communities" of variables
        num_communities = max(2, self.num_vars // 5)
        community_size = self.num_vars // num_communities
        communities = []
        
        for i in range(num_communities):
            start = i * community_size + 1
            end = min((i + 1) * community_size + 1, self.num_vars + 1)
            communities.append(list(range(start, end)))
        
        # Generate clauses with bias towards same community
        for _ in range(self.num_clauses):
            # Variable clause length (1 to clause_length)
            clause_len = np.random.randint(1, self.clause_length + 1)
            
            # 70% chance to pick from same community, 30% chance to mix
            if np.random.random() < 0.7 and len(communities) > 0:
                # Pick from same community
                community = np.random.choice(len(communities))
                available_vars = communities[community]
                if len(available_vars) >= clause_len:
                    vars_in_clause = np.random.choice(
                        available_vars, size=clause_len, replace=False
                    )
                else:
                    # Fallback to random selection
                    vars_in_clause = np.random.choice(
                        range(1, self.num_vars + 1), size=clause_len, replace=False
                    )
            else:
                # Mix variables from different communities
                vars_in_clause = np.random.choice(
                    range(1, self.num_vars + 1), size=clause_len, replace=False
                )
            
            # Random polarities with slight bias towards positive
            clause = []
            for var in vars_in_clause:
                if np.random.random() < 0.4:  # 40% negative, 60% positive
                    clause.append(-var)
                else:
                    clause.append(var)
            
            clauses.append(clause)
        
        # Create SAT task
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        return sat_task
    
    def generate_satisfiable_instance(
        self, 
        num_vars: int, 
        num_clauses: Optional[int] = None,
        max_attempts: int = 100
    ) -> SATTask:
        """Generate a guaranteed satisfiable SAT instance."""
        # Use planted solution method for guaranteed satisfiability
        old_type = self.distribution_type
        self.distribution_type = LOGIC_TYPE.PLANTED
        
        task = self.generate(num_vars=num_vars, num_clauses=num_clauses)
        
        self.distribution_type = old_type  # Restore original type
        return task
    
    def generate_unsatisfiable_instance(
        self,
        num_vars: int,
        num_clauses: Optional[int] = None,
        max_attempts: int = 100
    ) -> SATTask:
        """Generate a guaranteed unsatisfiable SAT instance."""
        # Add contradictory clauses to ensure unsatisfiability
        # Start with a random instance
        task = self.generate(num_vars=num_vars, num_clauses=num_clauses)
        
        # Add contradictory unit clauses for first few variables
        additional_clauses = []
        for i in range(min(3, num_vars)):
            var = i + 1
            # Add both positive and negative unit clauses for the same variable
            additional_clauses.append([var])
            additional_clauses.append([-var])
        
        # Combine original and contradictory clauses
        all_clauses = task.clauses + additional_clauses
        
        # Create new unsatisfiable task
        unsat_task = SATTask(precision=self.precision)
        unsat_task.from_data(clauses=all_clauses, num_vars=num_vars)
        return unsat_task