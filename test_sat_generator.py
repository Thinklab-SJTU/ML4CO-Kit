#!/usr/bin/env python3
"""
Test script for SAT Generator implementation.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from ml4co_kit.generator.logic.sat import SATGenerator, LOGIC_TYPE


def test_sat_generator_basic():
    """Test basic SAT generator functionality."""
    print("=== Testing SAT Generator Basic Functionality ===")
    
    generator = SATGenerator(
        distribution_type=LOGIC_TYPE.UNIFORM_RANDOM,
        num_vars=5,
        num_clauses=15,
        clause_length=3,
        seed=42
    )
    
    # Generate instance
    sat_task = generator.generate()
    
    print(f"Generated SAT instance:")
    print(f"  Variables: {sat_task.num_vars}")
    print(f"  Clauses: {sat_task.get_num_clauses()}")
    print(f"  Clause data: {sat_task.clauses[:3]}...")  # Show first 3 clauses
    
    # Test with random assignment
    assignment = np.random.choice([0, 1], size=sat_task.num_vars)
    satisfied = sat_task.evaluate(assignment)
    print(f"  Random assignment {assignment} satisfies {satisfied} clauses")
    
    return True


def test_sat_generator_distributions():
    """Test different SAT generation distributions."""
    print("\n=== Testing Different SAT Distributions ===")
    
    num_vars = 8
    num_clauses = 20
    
    distributions = [
        LOGIC_TYPE.RANDOM,
        LOGIC_TYPE.UNIFORM_RANDOM,
        LOGIC_TYPE.PLANTED,
        LOGIC_TYPE.PHASE_TRANSITION,
        LOGIC_TYPE.INDUSTRIAL
    ]
    
    for dist_type in distributions:
        print(f"\nTesting {dist_type} distribution:")
        
        generator = SATGenerator(
            distribution_type=dist_type,
            num_vars=num_vars,
            num_clauses=num_clauses,
            clause_length=3,
            seed=42
        )
        
        sat_task = generator.generate()
        print(f"  Generated {sat_task.get_num_clauses()} clauses for {sat_task.num_vars} variables")
        
        # Test planted solution has reference
        if dist_type == LOGIC_TYPE.PLANTED and sat_task.ref_assignment is not None:
            ref_satisfied = sat_task.evaluate(sat_task.ref_assignment.astype(np.int32))
            print(f"  Planted solution satisfies {ref_satisfied} clauses")
        
        # Show clause length distribution
        clause_lengths = [len(clause) for clause in sat_task.clauses]
        avg_length = np.mean(clause_lengths)
        print(f"  Average clause length: {avg_length:.2f}")
    
    return True


def test_sat_generator_special():
    """Test special SAT generation methods."""
    print("\n=== Testing Special SAT Generation ===")
    
    generator = SATGenerator(seed=42)
    
    # Test satisfiable instance generation
    print("Generating satisfiable instance:")
    sat_instance = generator.generate_satisfiable_instance(num_vars=6, num_clauses=15)
    print(f"  Variables: {sat_instance.num_vars}, Clauses: {sat_instance.get_num_clauses()}")
    
    if sat_instance.ref_assignment is not None:
        ref_satisfied = sat_instance.evaluate(sat_instance.ref_assignment.astype(np.int32))
        is_sat = sat_instance.is_satisfiable(sat_instance.ref_assignment.astype(np.int32))
        print(f"  Reference solution satisfies {ref_satisfied} clauses, SAT: {is_sat}")
    
    # Test unsatisfiable instance generation
    print("\nGenerating unsatisfiable instance:")
    unsat_instance = generator.generate_unsatisfiable_instance(num_vars=6, num_clauses=15)
    print(f"  Variables: {unsat_instance.num_vars}, Clauses: {unsat_instance.get_num_clauses()}")
    
    # Try all possible assignments for small instance to verify unsatisfiability
    if unsat_instance.num_vars <= 4:  # Only for very small instances
        all_unsat = True
        for i in range(2**unsat_instance.num_vars):
            assignment = np.array([
                (i >> j) & 1 for j in range(unsat_instance.num_vars)
            ])
            if unsat_instance.is_satisfiable(assignment):
                all_unsat = False
                break
        print(f"  Verified unsatisfiable: {all_unsat}")
    
    return True


def test_sat_generator_phase_transition():
    """Test phase transition generation."""
    print("\n=== Testing Phase Transition Generation ===")
    
    generator = SATGenerator(
        distribution_type=LOGIC_TYPE.PHASE_TRANSITION,
        clause_length=3,
        seed=42
    )
    
    # Test different variable counts
    for num_vars in [10, 20, 30]:
        sat_task = generator.generate(num_vars=num_vars)
        ratio = sat_task.get_num_clauses() / sat_task.num_vars
        print(f"  {num_vars} vars: {sat_task.get_num_clauses()} clauses, ratio: {ratio:.2f}")
    
    return True


def test_sat_generator_parameters():
    """Test parameter handling and edge cases."""
    print("\n=== Testing Parameter Handling ===")
    
    # Test auto clause calculation
    generator = SATGenerator(clause_length=3)
    for num_vars in [5, 10, 20]:
        sat_task = generator.generate(num_vars=num_vars)
        ratio = sat_task.get_num_clauses() / sat_task.num_vars
        print(f"  {num_vars} vars: auto-calculated {sat_task.get_num_clauses()} clauses (ratio: {ratio:.2f})")
    
    # Test different k-SAT
    print("\nTesting different k-SAT:")
    for k in [2, 3, 4, 5]:
        generator = SATGenerator(clause_length=k, seed=42)
        sat_task = generator.generate(num_vars=10)
        clause_lengths = [len(clause) for clause in sat_task.clauses]
        print(f"  {k}-SAT: clause lengths {min(clause_lengths)}-{max(clause_lengths)}")
    
    return True


def test_sat_generator_reproducibility():
    """Test reproducibility with seeds."""
    print("\n=== Testing Reproducibility ===")
    
    # Generate two instances with same seed
    generator1 = SATGenerator(seed=123)
    task1 = generator1.generate(num_vars=5, num_clauses=10)
    
    generator2 = SATGenerator(seed=123)
    task2 = generator2.generate(num_vars=5, num_clauses=10)
    
    # Check if they're identical
    same_clauses = (task1.clauses == task2.clauses)
    print(f"  Same seed produces identical instances: {same_clauses}")
    
    # Generate with different seed
    generator3 = SATGenerator(seed=456)
    task3 = generator3.generate(num_vars=5, num_clauses=10)
    
    different_clauses = (task1.clauses != task3.clauses)
    print(f"  Different seeds produce different instances: {different_clauses}")
    
    return True


if __name__ == "__main__":
    print("Testing SAT Generator Implementation")
    print("=" * 50)
    
    try:
        # Run all tests
        test_sat_generator_basic()
        test_sat_generator_distributions()
        test_sat_generator_special()
        test_sat_generator_phase_transition()
        test_sat_generator_parameters()
        test_sat_generator_reproducibility()
        
        print("\n" + "=" * 50)
        print("✅ All SAT Generator tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)