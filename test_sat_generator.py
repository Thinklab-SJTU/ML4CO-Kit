#!/usr/bin/env python3
"""
Test script for SAT Generator implementation.

This script tests the SAT problem generator with various distributions
and parameters to ensure correct functionality.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path so we can import ml4co_kit
sys.path.insert(0, os.path.dirname(__file__))

from ml4co_kit.generator.logic.sat import SATGenerator
from ml4co_kit.task.logic.sat import SATTask


def test_sat_generator_basic():
    """Test basic SAT generator functionality."""
    print("=== Testing SAT Generator Basic Functionality ===")
    
    generator = SATGenerator()
    
    # Test basic generation
    task = generator.generate(num_vars=10, num_clauses=20, distribution='RANDOM', seed=42)
    
    print(f"Generated SAT instance:")
    print(f"Variables: {task.num_vars}")
    print(f"Clauses: {task.get_num_clauses()}")
    print(f"Task type: {task.task_type}")
    
    # Basic assertions
    assert task.num_vars == 10
    assert task.get_num_clauses() == 20
    assert len(task.clauses) == 20
    print("✅ Basic generator test passed")


def test_sat_generator_distributions():
    """Test different SAT generation distributions."""
    print("\n=== Testing SAT Generator Distributions ===")
    
    generator = SATGenerator()
    num_vars = 8
    num_clauses = 16
    
    distributions = ['RANDOM', 'UNIFORM_RANDOM', 'PLANTED', 'PHASE_TRANSITION', 'INDUSTRIAL']
    
    for dist in distributions:
        print(f"Testing distribution: {dist}")
        task = generator.generate(
            num_vars=num_vars, 
            num_clauses=num_clauses, 
            distribution=dist, 
            seed=123
        )
        
        assert task.num_vars == num_vars
        assert task.get_num_clauses() == num_clauses
        
        # Check that clauses contain valid literals
        for clause in task.clauses:
            for literal in clause:
                assert abs(literal) <= num_vars
                assert literal != 0
        
        print(f"  ✅ {dist} distribution test passed")


def test_sat_generator_special():
    """Test special cases for SAT generation."""
    print("\n=== Testing SAT Generator Special Cases ===")
    
    generator = SATGenerator()
    
    # Test small instance
    small_task = generator.generate(num_vars=3, num_clauses=5, distribution='RANDOM', seed=1)
    assert small_task.num_vars == 3
    assert small_task.get_num_clauses() == 5
    print("✅ Small instance test passed")
    
    # Test larger instance  
    large_task = generator.generate(num_vars=50, num_clauses=200, distribution='RANDOM', seed=2)
    assert large_task.num_vars == 50
    assert large_task.get_num_clauses() == 200
    print("✅ Large instance test passed")
    
    # Test planted solution distribution
    planted_task = generator.generate(num_vars=10, num_clauses=30, distribution='PLANTED', seed=3)
    assert planted_task.num_vars == 10
    assert planted_task.get_num_clauses() == 30
    print("✅ Planted solution test passed")


def test_sat_generator_phase_transition():
    """Test phase transition region generation."""
    print("\n=== Testing SAT Generator Phase Transition ===")
    
    generator = SATGenerator()
    
    # Phase transition around ratio 4.3 for 3-SAT
    num_vars = 20
    num_clauses = int(4.3 * num_vars)  # ~86 clauses
    
    task = generator.generate(
        num_vars=num_vars, 
        num_clauses=num_clauses, 
        distribution='PHASE_TRANSITION', 
        seed=456
    )
    
    assert task.num_vars == num_vars
    assert task.get_num_clauses() == num_clauses
    
    # Check clause structure (should be 3-SAT for phase transition)
    clause_lengths = [len(clause) for clause in task.clauses]
    avg_clause_length = np.mean(clause_lengths)
    
    print(f"Average clause length: {avg_clause_length:.2f}")
    print(f"Clause/variable ratio: {num_clauses/num_vars:.2f}")
    
    assert 2.5 <= avg_clause_length <= 3.5  # Should be close to 3 for 3-SAT
    print("✅ Phase transition test passed")


def test_sat_generator_parameters():
    """Test SAT generator parameter validation."""
    print("\n=== Testing SAT Generator Parameters ===")
    
    generator = SATGenerator()
    
    # Test different clause lengths
    for clause_len in [2, 3, 4, 5]:
        task = generator.generate(
            num_vars=10, 
            num_clauses=20, 
            distribution='UNIFORM_RANDOM',
            clause_length=clause_len,
            seed=789
        )
        
        # Check that clauses have approximately the right length
        clause_lengths = [len(clause) for clause in task.clauses]
        avg_length = np.mean(clause_lengths)
        
        print(f"Target clause length: {clause_len}, actual average: {avg_length:.2f}")
        assert abs(avg_length - clause_len) < 1.0  # Allow some variation
    
    print("✅ Parameter validation test passed")


def test_sat_generator_reproducibility():
    """Test that SAT generation is reproducible with same seed."""
    print("\n=== Testing SAT Generator Reproducibility ===")
    
    generator = SATGenerator()
    
    # Generate twice with same seed
    task1 = generator.generate(num_vars=15, num_clauses=45, distribution='RANDOM', seed=999)
    task2 = generator.generate(num_vars=15, num_clauses=45, distribution='RANDOM', seed=999)
    
    # Should produce identical instances
    assert task1.num_vars == task2.num_vars
    assert task1.get_num_clauses() == task2.get_num_clauses()
    assert task1.clauses == task2.clauses
    
    print("✅ Reproducibility test passed")
    
    # Generate with different seed - should be different
    task3 = generator.generate(num_vars=15, num_clauses=45, distribution='RANDOM', seed=1000)
    assert task3.clauses != task1.clauses
    
    print("✅ Different seed produces different instance")


def main():
    """Run all SAT generator tests."""
    print("🚀 SAT Generator Testing")
    print("=" * 50)
    
    try:
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


if __name__ == "__main__":
    main()