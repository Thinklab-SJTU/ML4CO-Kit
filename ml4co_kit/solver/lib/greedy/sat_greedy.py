r"""
Greedy Solver for SAT (Boolean Satisfiability Problem).

This module implements a greedy heuristic solver using frequency-based 
variable selection and unit propagation.
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
from typing import List, Set, Dict, Optional, Tuple
from ml4co_kit.task.logic.sat import SATTask


def sat_greedy(task_data: SATTask, max_iterations: int = 1000) -> None:
    
    # Extract problem data
    n_vars = task_data.n_vars
    clauses = [list(clause) for clause in task_data.cnf]  # Mutable copy
    
    # Initialize assignment state
    assignment = {}  # var -> bool (1-indexed variables)
    satisfied_clauses = set()  # Indices of satisfied clauses
    
    def evaluate_clause(clause_idx: int) -> bool:
        """Check if a clause is satisfied by current assignment."""
        clause = clauses[clause_idx]
        for literal in clause:
            var = abs(literal)
            if var in assignment:
                # Check if this literal is satisfied
                if (literal > 0 and assignment[var]) or (literal < 0 and not assignment[var]):
                    return True
        return False
    
    def get_unit_clauses() -> List[int]:
        """Find unit clauses (only one unassigned literal)."""
        unit_literals = []
        for i, clause in enumerate(clauses):
            if i in satisfied_clauses:
                continue
                
            unassigned_literals = []
            for literal in clause:
                var = abs(literal)
                if var not in assignment:
                    unassigned_literals.append(literal)
            
            if len(unassigned_literals) == 1:
                unit_literals.append(unassigned_literals[0])
            elif len(unassigned_literals) == 0:
                # Conflict: clause not satisfied and no unassigned literals
                return None  # Indicates conflict
                
        return unit_literals
    
    def unit_propagation() -> bool:
        """
        Apply unit propagation to infer forced assignments.
        Returns True if successful, False if conflict detected.
        """
        while True:
            unit_literals = get_unit_clauses()
            if unit_literals is None:
                return False  # Conflict detected
            if not unit_literals:
                break  # No more unit clauses
                
            # Assign unit literals
            for literal in unit_literals:
                var = abs(literal)
                value = literal > 0
                if var in assignment and assignment[var] != value:
                    return False  # Conflict: variable already assigned differently
                assignment[var] = value
            
            # Update satisfied clauses
            update_satisfied_clauses()
        
        return True
    
    def update_satisfied_clauses():
        """Update the set of satisfied clauses."""
        satisfied_clauses.clear()
        for i in range(len(clauses)):
            if evaluate_clause(i):
                satisfied_clauses.add(i)
    
    def calculate_variable_frequencies() -> Dict[int, Tuple[int, int]]:
        """
        Calculate positive and negative frequencies for unassigned variables.
        Returns dict: var -> (positive_freq, negative_freq)
        """
        frequencies = {}
        
        for i, clause in enumerate(clauses):
            if i in satisfied_clauses:
                continue
                
            for literal in clause:
                var = abs(literal)
                if var not in assignment:
                    if var not in frequencies:
                        frequencies[var] = [0, 0]  # [positive, negative]
                    
                    if literal > 0:
                        frequencies[var][0] += 1
                    else:
                        frequencies[var][1] += 1
        
        return {var: tuple(freqs) for var, freqs in frequencies.items()}
    
    def select_next_variable() -> Optional[Tuple[int, bool]]:
        """
        Select next variable and polarity using greedy heuristic.
        Returns (variable, polarity) or None if no unassigned variables.
        """
        frequencies = calculate_variable_frequencies()
        if not frequencies:
            return None
        
        # Find variable with highest total frequency
        best_var = None
        best_score = -1
        best_polarity = True
        
        for var, (pos_freq, neg_freq) in frequencies.items():
            total_freq = pos_freq + neg_freq
            if total_freq > best_score:
                best_score = total_freq
                best_var = var
                # Choose polarity with higher frequency
                best_polarity = pos_freq >= neg_freq
        
        return (best_var, best_polarity)
    
    # Main solving loop
    iteration = 0
    while iteration < max_iterations:
        # Apply unit propagation
        if not unit_propagation():
            # Conflict detected - instance likely unsatisfiable with current approach
            print("Greedy SAT solver detected conflict - instance may be unsatisfiable")
            task_data.sol = None
            return
        
        # Check if all clauses are satisfied
        if len(satisfied_clauses) == len(clauses):
            # Found satisfying assignment
            solution = np.zeros(n_vars, dtype=np.int32)
            for var in range(1, n_vars + 1):
                if var in assignment:
                    solution[var - 1] = 1 if assignment[var] else 0
                else:
                    # Unassigned variables can be set arbitrarily
                    solution[var - 1] = 0
            
            task_data.sol = solution
            return
        
        # Select next variable to assign
        next_assignment = select_next_variable()
        if next_assignment is None:
            # No unassigned variables but not all clauses satisfied
            print("Greedy SAT solver: no unassigned variables but unsatisfied clauses remain")
            task_data.sol = None
            return
        
        var, polarity = next_assignment
        assignment[var] = polarity
        update_satisfied_clauses()
        
        iteration += 1
    
    # Maximum iterations reached without solution
    print(f"Greedy SAT solver reached maximum iterations ({max_iterations})")
    task_data.sol = None