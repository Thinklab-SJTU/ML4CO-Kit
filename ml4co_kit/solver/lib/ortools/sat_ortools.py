r"""
OR-Tools Solver for SAT (Boolean Satisfiability Problem)
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