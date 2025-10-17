r"""
OR-Tools Solver for SAT (Boolean Satisfiability Problem)

This module implements an OR-Tools CP-SAT based solver for the SAT problem using Constraint Programming.

Research Background & References:
================================

1. Constraint Programming (CP) for SAT:
   - Rossi, F., Van Beek, P., & Walsh, T. (Eds.). (2006). "Handbook of constraint programming". 
     Elsevier. Chapter 26: "Satisfiability".
   - CP approaches to Boolean satisfiability with constraint propagation and search
   - Integration of SAT solving within constraint programming frameworks

2. OR-Tools CP-SAT Solver:
   - Perron, L., & Furnon, V. (2019). "OR-Tools". Google.
     https://developers.google.com/optimization/
   - Modern constraint programming solver with SAT-based backend
   - Combines conflict-driven clause learning (CDCL) with constraint propagation

3. CDCL (Conflict-Driven Clause Learning):
   - Silva, J. P. M., & Sakallah, K. A. (1999). "GRASP: A search algorithm for propositional satisfiability". 
     IEEE Transactions on Computers, 48(5), 506-521.
   - Modern SAT solving algorithm with backjumping and clause learning
   - Foundation for most competitive SAT solvers including CP-SAT backend

4. SAT Competition and DIMACS:
   - SAT Competition. (2023). "International SAT Competition". 
     http://www.satcompetition.org/
   - Annual competition driving SAT solver development and standardization
   - DIMACS CNF format as de facto standard for SAT instances

5. Boolean Constraint Satisfaction:
   - Tsang, E. (1993). "Foundations of constraint satisfaction". 
     Academic Press. Chapter 12.
   - Theoretical foundations of Boolean constraint satisfaction problems
   - Relationship between SAT and general constraint satisfaction

Mathematical Formulation:
========================

Given a SAT instance in CNF with Boolean variables x₁, x₂, ..., xₙ and clauses C₁, C₂, ..., Cₘ:

CP Variables:
- xᵢ ∈ {False, True} for i = 1, ..., n (Boolean domain variables)

CP Constraints:
For each clause Cⱼ = (l₁ ∨ l₂ ∨ ... ∨ lₖ):
- AddBoolOr([l₁, l₂, ..., lₖ]) where each lᵢ is either xᵤ or xᵤ.Not()

Constraint Propagation:
- Unit propagation: if clause has only one unassigned literal, assign it True
- Pure literal elimination: if variable appears only positively/negatively, assign appropriately
- Conflict analysis: when contradiction found, learn new clause to prevent same conflict

Algorithm Details:
==================

1. Parse CNF formula and create Boolean variables
2. Add BoolOr constraint for each clause
3. Apply constraint propagation and search with CDCL
4. Extract satisfying assignment or report UNSAT

Time Complexity: NP-complete (exponential worst-case)
Space Complexity: O(n + m + learned_clauses)
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
from ortools.sat.python import cp_model
from ml4co_kit.task.logic.sat import SATTask


def sat_ortools(
    task_data: SATTask, 
    ortools_time_limit: int = 10
):
    """
    Solve SAT problem using OR-Tools CP-SAT solver.
    
    This function uses Google's OR-Tools CP-SAT solver, which implements a modern
    Conflict-Driven Clause Learning (CDCL) algorithm with constraint propagation.
    The solver combines techniques from both SAT solving and constraint programming.
    
    Args:
        task_data (SATTask): SAT instance containing CNF formula
        ortools_time_limit (int): Maximum solving time in seconds (default: 10)
    
    Returns:
        None: Solution is stored directly in task_data.solution
        
    CP Model:
        Variables: xᵢ ∈ {False, True} for i = 1, ..., n_vars
        Constraints: For each clause C = (l₁ ∨ l₂ ∨ ... ∨ lₖ):
                    AddBoolOr([l₁, l₂, ..., lₖ])
        
    Example CNF to CP Transformation:
        Clause (x₁ ∨ ¬x₂ ∨ x₃) becomes:
        AddBoolOr([x[0], x[1].Not(), x[2]]) 
        (using 0-based indexing)
    """
    
    # Extract problem data
    n_vars = task_data.n_vars
    clauses = task_data.cnf
    
    # Create CP-SAT model
    model = cp_model.CpModel()
    
    # Create Boolean variables for each SAT variable
    # x[i] represents the Boolean assignment for variable (i+1) in the CNF
    x = [model.NewBoolVar(f'x_{i+1}') for i in range(n_vars)]
    
    # Add constraints for each clause
    for clause_idx, clause in enumerate(clauses):
        # Convert clause to list of Boolean literals for OR constraint
        bool_literals = []
        
        for literal in clause:
            if literal > 0:
                # Positive literal: x[literal-1]
                var_idx = literal - 1
                bool_literals.append(x[var_idx])
            else:
                # Negative literal: ¬x[|literal|-1] = x[|literal|-1].Not()
                var_idx = abs(literal) - 1
                bool_literals.append(x[var_idx].Not())
        
        # Add disjunction constraint: at least one literal must be True
        model.AddBoolOr(bool_literals)
    
    # Create and configure solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = ortools_time_limit
    
    # For reproducibility and performance tuning
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1  # Single-threaded for consistency
    
    # Solve the model
    status = solver.Solve(model)
    
    # Process solution based on solver status
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract Boolean assignment from solution
        solution = np.array([
            1 if solver.BooleanValue(x[i]) else 0 
            for i in range(n_vars)
        ], dtype=np.int32)
        
        # Store solution in task data
        task_data.sol = solution
        
        # Optional: verify solution correctness
        if hasattr(task_data, '_validate_solution'):
            is_valid = task_data._validate_solution(solution)
            if not is_valid:
                print(f"Warning: OR-Tools found invalid solution")
                
    elif status == cp_model.INFEASIBLE:
        # Instance is unsatisfiable
        print(f"SAT instance is UNSATISFIABLE (OR-Tools)")
        task_data.sol = None
        
    elif status == cp_model.MODEL_INVALID:
        # Model formulation error
        raise ValueError(f"Invalid CP model formulation for SAT instance")
        
    elif status == cp_model.UNKNOWN:
        # Time limit or other termination without definitive result
        print(f"OR-Tools solver terminated with UNKNOWN status")
        task_data.sol = None
        
    else:
        # Other status codes
        print(f"OR-Tools solver status: {solver.StatusName(status)}")
        task_data.sol = None
        
    # Optional: print solving statistics for debugging
    if hasattr(task_data, 'name') and task_data.name:
        print(f"OR-Tools SAT solving stats for {task_data.name}:")
        print(f"  Status: {solver.StatusName(status)}")
        print(f"  Wall time: {solver.WallTime():.3f}s")
        print(f"  Branches: {solver.NumBranches()}")
        print(f"  Conflicts: {solver.NumConflicts()}")
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"  Solution found: {n_vars} variables assigned")