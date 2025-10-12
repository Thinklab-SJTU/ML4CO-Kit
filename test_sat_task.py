#!/usr/bin/env python3
"""
Test script for SAT Task implementation.
"""

import numpy as np
import pathlib
import sys
import os

# Add the parent directory to the path so we can import ml4co_kit
sys.path.insert(0, os.path.dirname(__file__))

# Direct imports to avoid dependency issues
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.logic.sat import SATTask


def test_sat_task_basic():
    """Test basic SAT task functionality."""
    print("=== Testing SAT Task Basic Functionality ===")
    
    # Create a simple SAT instance
    # Formula: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [
        [1, 2],      # x1 OR x2
        [-1, 3],     # NOT x1 OR x3  
        [-2, -3]     # NOT x2 OR NOT x3
    ]
    
    sat_task = SATTask()
    sat_task.from_data(clauses=clauses, num_vars=3)
    
    print(f"Task type: {sat_task.task_type}")
    print(f"Number of variables: {sat_task.num_vars}")
    print(f"Number of clauses: {sat_task.get_num_clauses()}")
    print(f"Clauses: {sat_task.clauses}")
    
    # Test satisfying assignment: x1=True, x2=False, x3=True
    assignment1 = np.array([1, 0, 1])  # [x1=True, x2=False, x3=True]
    sat_task.set_assignment(assignment1)
    
    print(f"\nTesting assignment {assignment1}:")
    print(f"Is valid: {sat_task.check_constraints(assignment1)}")
    satisfied_clauses = sat_task.evaluate(assignment1)
    print(f"Satisfied clauses: {satisfied_clauses}/{len(clauses)}")
    print(f"Is satisfiable: {sat_task.is_satisfiable(assignment1)}")
    print(f"Satisfaction ratio: {sat_task.get_clause_satisfaction_ratio(assignment1):.2f}")
    
    # Test unsatisfying assignment: x1=False, x2=False, x3=False
    assignment2 = np.array([0, 0, 0])
    print(f"\nTesting assignment {assignment2}:")
    satisfied_clauses2 = sat_task.evaluate(assignment2)
    print(f"Satisfied clauses: {satisfied_clauses2}/{len(clauses)}")
    print(f"Is satisfiable: {sat_task.is_satisfiable(assignment2)}")
    unsatisfied = sat_task.get_unsatisfied_clauses(assignment2)
    print(f"Unsatisfied clause indices: {unsatisfied}")
    
    return True


def test_sat_task_dimacs():
    """Test DIMACS file format support."""
    print("\n=== Testing DIMACS Format Support ===")
    
    # Create a test DIMACS file
    dimacs_content = """c This is a test SAT instance
c 3 variables, 3 clauses
p cnf 3 3
1 2 0
-1 3 0
-2 -3 0
"""
    
    test_file = pathlib.Path("test_sat.cnf")
    with open(test_file, 'w') as f:
        f.write(dimacs_content)
    
    # Load from DIMACS
    sat_task = SATTask()
    sat_task.from_dimacs(test_file)
    
    print(f"Loaded from DIMACS:")
    print(f"Variables: {sat_task.num_vars}")
    print(f"Clauses: {sat_task.get_num_clauses()}")
    print(f"Clause data: {sat_task.clauses}")
    
    # Test satisfying assignment
    assignment = np.array([1, 0, 1])
    sat_task.set_assignment(assignment)
    satisfied = sat_task.evaluate(assignment)
    print(f"Assignment {assignment} satisfies {satisfied} clauses")
    
    # Save back to DIMACS
    output_file = pathlib.Path("test_output.cnf")
    sat_task.to_dimacs(output_file)
    print(f"Saved to {output_file}")
    
    # Clean up
    test_file.unlink()
    output_file.unlink()
    
    return True


def test_sat_task_evaluation():
    """Test detailed evaluation functionality."""
    print("\n=== Testing SAT Evaluation ===")
    
    # Create a more complex SAT instance
    # (x1 OR x2 OR x3) AND (x1 OR NOT x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [
        [1, 2, 3],    # x1 OR x2 OR x3
        [1, -2],      # x1 OR NOT x2
        [-1, 3],      # NOT x1 OR x3
        [-2, -3]      # NOT x2 OR NOT x3
    ]
    
    sat_task = SATTask()
    sat_task.from_data(clauses=clauses, num_vars=3)
    
    print(f"Testing complex SAT instance with {len(clauses)} clauses")
    
    # Test all possible assignments for 3 variables
    for i in range(8):  # 2^3 = 8 combinations
        assignment = np.array([
            (i >> 2) & 1,  # x1
            (i >> 1) & 1,  # x2
            i & 1          # x3
        ])
        
        satisfied = sat_task.evaluate(assignment)
        is_sat = sat_task.is_satisfiable(assignment)
        
        print(f"Assignment {assignment}: {satisfied}/4 clauses satisfied, SAT: {is_sat}")
    
    return True


def test_sat_task_copy():
    """Test SAT task copying functionality."""
    print("\n=== Testing SAT Task Copy ===")
    
    clauses = [[1, -2], [2, 3], [-1, -3]]
    original = SATTask()
    original.from_data(clauses=clauses, num_vars=3)
    original.set_assignment(np.array([1, 0, 1]))
    
    # Test copy
    copied = original.copy()
    
    print(f"Original: {original}")
    print(f"Copied: {copied}")
    print(f"Same clauses: {original.clauses == copied.clauses}")
    print(f"Same assignment: {np.array_equal(original.assignment, copied.assignment)}")
    
    # Modify copy and ensure original is unchanged
    copied.set_assignment(np.array([0, 1, 0]))
    print(f"After modifying copy:")
    print(f"Original assignment: {original.assignment}")
    print(f"Copied assignment: {copied.assignment}")
    
    return True


if __name__ == "__main__":
    print("Testing SAT Task Implementation")
    print("=" * 50)
    
    try:
        # Run all tests
        test_sat_task_basic()
        test_sat_task_dimacs() 
        test_sat_task_evaluation()
        test_sat_task_copy()
        
        print("\n" + "=" * 50)
        print("✅ All SAT Task tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)