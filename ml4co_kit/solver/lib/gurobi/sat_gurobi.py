r"""
Gurobi Solver for SAT (Boolean Satisfiability Problem).

Implements ILP formulation for SAT using Gurobi optimizer.
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
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from ml4co_kit.task.logic.sat import SATTask


def sat_gurobi(
    task_data: SATTask,
    gurobi_time_limit: float = 10.0
):
    """
    Solve SAT problem using Gurobi ILP solver.
    
    This function implements the Integer Linear Programming (ILP) formulation for the Boolean
    Satisfiability Problem (SAT). Each clause in CNF is converted to a linear constraint
    ensuring at least one literal in the clause is satisfied.
    
    Args:
        task_data (SATTask): SAT instance containing CNF formula
        gurobi_time_limit (float): Maximum solving time in seconds (default: 10.0)
    
    Returns:
        None: Solution is stored directly in task_data.solution
        
    Mathematical Model:
        Variables: xᵢ ∈ {0, 1} for i = 1, ..., n_vars
        Constraints: For each clause C = (l₁ ∨ l₂ ∨ ... ∨ lₖ):
                    Σ(satisfied literals) ≥ 1
        
    Example CNF to ILP Transformation:
        Clause (x₁ ∨ ¬x₂ ∨ x₃) becomes:
        x₁ + (1 - x₂) + x₃ ≥ 1
        which simplifies to: x₁ - x₂ + x₃ ≥ 0
    """
    
    # Extract problem data
    n_vars = task_data.n_vars
    clauses = task_data.cnf
    
    # Create Gurobi model
    model_name = f"SAT-{task_data.name}" if hasattr(task_data, 'name') else "SAT"
    model = gp.Model(model_name)
    
    # Configure solver parameters
    model.setParam("OutputFlag", 0)  # Suppress output for clean execution
    model.setParam("TimeLimit", gurobi_time_limit)
    model.setParam("Threads", 1)  # Single-threaded for reproducibility
    
    # Create binary variables for each Boolean variable
    # x[i] = 1 if variable i+1 is True, 0 if False
    x = model.addVars(n_vars, vtype=GRB.BINARY, name="x")
    
    # Add constraints for each clause
    for clause_idx, clause in enumerate(clauses):
        # Build linear expression for clause satisfaction
        # For positive literal i: add x[i-1] 
        # For negative literal -i: add (1 - x[i-1]) = 1 - x[i-1]
        
        clause_expr = gp.LinExpr()  # Linear expression for current clause
        constant_term = 0  # Constant term from negative literals
        
        for literal in clause:
            if literal > 0:
                # Positive literal: x[literal-1]
                clause_expr.add(x[literal - 1], 1.0)
            else:
                # Negative literal: (1 - x[|literal|-1])
                # This contributes +1 to constant and -1 coefficient to variable
                var_idx = abs(literal) - 1
                clause_expr.add(x[var_idx], -1.0)
                constant_term += 1
        
        # Add constraint: clause_expr + constant_term >= 1
        # Equivalent to: clause_expr >= 1 - constant_term
        model.addConstr(
            clause_expr >= 1 - constant_term, 
            name=f"clause_{clause_idx}"
        )
    
    # Set objective (feasibility problem - any valid solution is acceptable)
    # For pure SAT: no objective needed, just find feasible solution
    # For MAX-SAT variant: could maximize satisfied clauses
    model.setObjective(0, GRB.MINIMIZE)  # Dummy objective for feasibility
    
    # Write model to file for debugging (optional)
    lp_filename = f"{model_name}.lp"
    model.write(lp_filename)
    
    # Solve the model
    model.optimize()
    
    # Clean up temporary file
    if os.path.exists(lp_filename):
        os.remove(lp_filename)
    
    # Extract and store solution
    if model.Status == GRB.OPTIMAL:
        # Extract variable assignments from optimal solution
        solution = np.array([int(x[i].X) for i in range(n_vars)], dtype=np.int32)
        task_data.sol = solution
        
        # Verify solution correctness (optional validation)
        if hasattr(task_data, '_validate_solution'):
            is_valid = task_data._validate_solution(solution)
            if not is_valid:
                print(f"Warning: Gurobi found invalid solution for {model_name}")
                
    elif model.Status == GRB.INFEASIBLE:
        # No satisfying assignment exists (UNSAT)
        print(f"SAT instance {model_name} is UNSATISFIABLE")
        task_data.sol = None
        
    elif model.Status == GRB.TIME_LIMIT:
        # Time limit reached without conclusive result
        print(f"Time limit reached for SAT instance {model_name}")
        task_data.sol = None
        
    else:
        # Other status (numerical issues, user interruption, etc.)
        print(f"Gurobi solver status: {model.Status} for {model_name}")
        task_data.sol = None