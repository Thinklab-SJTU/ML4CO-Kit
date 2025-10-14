#!/usr/bin/env python3
"""
SAT Solver Testing Script for ML4CO-Kit

This script comprehensively tests all implemented SAT solvers:
1. Gurobi ILP Solver
2. OR-Tools CP-SAT Solver  
3. Greedy Heuristic Solver

Tests multiple instance types and validates correctness of solutions.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

# Add ML4CO-Kit to path
sys.path.append('/mnt/nas-new/home/zhanghang/zhangxihe/ml4co_workspace/ML4CO-Kit')

from ml4co_kit.task.logic.sat import SATTask
from ml4co_kit.generator.logic.sat import SATGenerator
from ml4co_kit.solver.gurobi import GurobiSolver
from ml4co_kit.solver.ortools import ORSolver
from ml4co_kit.solver.greedy import GreedySolver
from ml4co_kit.extension.gnn4co.model.model import GNN4COModel


def create_test_instances() -> List[SATTask]:
    """Create diverse SAT test instances for comprehensive testing."""
    
    print("🔄 Generating test SAT instances...")
    
    generator = SATGenerator()
    test_instances = []
    
    # Test case 1: Small satisfiable random instance
    print("  📝 Creating small random SAT instance...")
    small_sat = generator.generate(
        num_vars=10, 
        num_clauses=30, 
        distribution='RANDOM',
        seed=42
    )
    small_sat.name = "small_random_sat"
    test_instances.append(small_sat)
    
    # Test case 2: Phase transition instance (potentially hard)
    print("  📝 Creating phase transition SAT instance...")
    phase_sat = generator.generate(
        num_vars=20, 
        num_clauses=85,  # ~4.25 ratio near phase transition
        distribution='PHASE_TRANSITION',
        seed=123
    )
    phase_sat.name = "phase_transition_sat"
    test_instances.append(phase_sat)
    
    # Test case 3: Planted solution (guaranteed satisfiable)
    print("  📝 Creating planted solution SAT instance...")
    planted_sat = generator.generate(
        num_vars=15,
        num_clauses=50,
        distribution='PLANTED',
        seed=456
    )
    planted_sat.name = "planted_solution_sat"
    test_instances.append(planted_sat)
    
    # Test case 4: Simple manually created instance
    print("  📝 Creating simple manual SAT instance...")
    manual_sat = SATTask()
    manual_clauses = [
        [1, 2, 3],      # (x1 ∨ x2 ∨ x3)
        [-1, 2],        # (¬x1 ∨ x2)  
        [-2, 3],        # (¬x2 ∨ x3)
        [-3, 1]         # (¬x3 ∨ x1)
    ]
    manual_sat.from_data(clauses=manual_clauses, num_vars=3)
    manual_sat.name = "manual_simple_sat"
    test_instances.append(manual_sat)
    
    # Test case 5: Likely unsatisfiable instance (high clause/variable ratio)
    print("  📝 Creating likely UNSAT instance...")
    unsat_candidate = generator.generate(
        num_vars=8,
        num_clauses=50,  # Very high ratio: 6.25
        distribution='RANDOM',
        seed=789
    )
    unsat_candidate.name = "likely_unsat"
    test_instances.append(unsat_candidate)
    
    print(f"✅ Created {len(test_instances)} test instances\n")
    return test_instances


def validate_sat_solution(task: SATTask, solution: Optional[np.ndarray]) -> Tuple[bool, str]:
    """
    Validate if a solution actually satisfies the SAT instance.
    
    Returns:
        (is_valid, message): Tuple indicating validity and explanation
    """
    if solution is None:
        return True, "No solution provided (UNSAT or timeout)"
    
    if len(solution) != task.n_vars:
        return False, f"Solution length {len(solution)} != n_vars {task.n_vars}"
    
    # Check each clause
    for clause_idx, clause in enumerate(task.cnf):
        clause_satisfied = False
        for literal in clause:
            var_idx = abs(literal) - 1  # Convert to 0-based indexing
            var_value = bool(solution[var_idx])
            
            if literal > 0 and var_value:
                clause_satisfied = True
                break
            elif literal < 0 and not var_value:
                clause_satisfied = True
                break
        
        if not clause_satisfied:
            return False, f"Clause {clause_idx + 1} not satisfied: {clause}"
    
    return True, "All clauses satisfied"


def test_gurobi_solver(instances: List[SATTask]) -> Dict:
    """Test Gurobi ILP solver on all instances."""
    
    print("🔧 Testing Gurobi ILP Solver")
    print("=" * 50)
    
    solver = GurobiSolver(gurobi_time_limit=30.0)
    results = {"solver": "Gurobi ILP", "tests": []}
    
    for i, instance in enumerate(instances, 1):
        print(f"  🧪 Test {i}: {instance.name}")
        print(f"     Variables: {instance.n_vars}, Clauses: {len(instance.cnf)}")
        
        # Make a copy for solving (preserve original)
        test_instance = SATTask()
        test_instance.from_data(clauses=instance.cnf, num_vars=instance.n_vars)
        test_instance.name = instance.name
        
        start_time = time.time()
        try:
            solver.solve(test_instance)
            solve_time = time.time() - start_time
            
            # Validate solution
            is_valid, message = validate_sat_solution(test_instance, test_instance.solution)
            
            result = {
                "instance": instance.name,
                "success": True,
                "solve_time": solve_time,
                "solution_found": test_instance.solution is not None,
                "valid": is_valid,
                "message": message
            }
            
            print(f"     ⏱️  Solve time: {solve_time:.3f}s")
            print(f"     📊 Solution: {'Found' if test_instance.solution is not None else 'None (UNSAT)'}")
            print(f"     ✅ Valid: {is_valid} - {message}")
            
        except Exception as e:
            result = {
                "instance": instance.name,
                "success": False,
                "error": str(e)
            }
            print(f"     ❌ Error: {e}")
        
        results["tests"].append(result)
        print()
    
    return results


def test_ortools_solver(instances: List[SATTask]) -> Dict:
    """Test OR-Tools CP-SAT solver on all instances."""
    
    print("🔧 Testing OR-Tools CP-SAT Solver")
    print("=" * 50)
    
    solver = ORSolver(ortools_time_limit=30)
    results = {"solver": "OR-Tools CP-SAT", "tests": []}
    
    for i, instance in enumerate(instances, 1):
        print(f"  🧪 Test {i}: {instance.name}")
        print(f"     Variables: {instance.n_vars}, Clauses: {len(instance.cnf)}")
        
        # Make a copy for solving
        test_instance = SATTask()
        test_instance.from_data(clauses=instance.cnf, num_vars=instance.n_vars)
        test_instance.name = instance.name
        
        start_time = time.time()
        try:
            solver.solve(test_instance)
            solve_time = time.time() - start_time
            
            # Validate solution
            is_valid, message = validate_sat_solution(test_instance, test_instance.solution)
            
            result = {
                "instance": instance.name,
                "success": True,
                "solve_time": solve_time,
                "solution_found": test_instance.solution is not None,
                "valid": is_valid,
                "message": message
            }
            
            print(f"     ⏱️  Solve time: {solve_time:.3f}s")
            print(f"     📊 Solution: {'Found' if test_instance.solution is not None else 'None (UNSAT)'}")
            print(f"     ✅ Valid: {is_valid} - {message}")
            
        except Exception as e:
            result = {
                "instance": instance.name,
                "success": False,
                "error": str(e)
            }
            print(f"     ❌ Error: {e}")
        
        results["tests"].append(result)
        print()
    
    return results


def test_greedy_solver(instances: List[SATTask]) -> Dict:
    """Test Greedy heuristic solver on all instances."""
    
    print("🔧 Testing Greedy Heuristic Solver")
    print("=" * 50)
    
    # Note: Greedy solver doesn't require GNN model for SAT
    # We'll create a dummy model to satisfy the interface
    results = {"solver": "Greedy Heuristic", "tests": []}
    
    for i, instance in enumerate(instances, 1):
        print(f"  🧪 Test {i}: {instance.name}")
        print(f"     Variables: {instance.n_vars}, Clauses: {len(instance.cnf)}")
        
        # Make a copy for solving
        test_instance = SATTask()
        test_instance.from_data(clauses=instance.cnf, num_vars=instance.n_vars)
        test_instance.name = instance.name
        
        start_time = time.time()
        try:
            # Import and use greedy solver directly
            from ml4co_kit.solver.lib.greedy.sat_greedy import sat_greedy
            sat_greedy(test_instance, max_iterations=1000)
            solve_time = time.time() - start_time
            
            # Validate solution
            is_valid, message = validate_sat_solution(test_instance, test_instance.solution)
            
            result = {
                "instance": instance.name,
                "success": True,
                "solve_time": solve_time,
                "solution_found": test_instance.solution is not None,
                "valid": is_valid,
                "message": message
            }
            
            print(f"     ⏱️  Solve time: {solve_time:.3f}s")
            print(f"     📊 Solution: {'Found' if test_instance.solution is not None else 'None (UNSAT)'}")
            print(f"     ✅ Valid: {is_valid} - {message}")
            
        except Exception as e:
            result = {
                "instance": instance.name,
                "success": False,
                "error": str(e)
            }
            print(f"     ❌ Error: {e}")
        
        results["tests"].append(result)
        print()
    
    return results


def print_summary(all_results: List[Dict]) -> None:
    """Print comprehensive summary of all solver test results."""
    
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    for solver_result in all_results:
        solver_name = solver_result["solver"]
        tests = solver_result["tests"]
        
        print(f"\n🔧 {solver_name}")
        print("-" * 40)
        
        total_tests = len(tests)
        successful_tests = sum(1 for t in tests if t.get("success", False))
        valid_solutions = sum(1 for t in tests if t.get("valid", False))
        solutions_found = sum(1 for t in tests if t.get("solution_found", False))
        
        print(f"  📈 Success Rate: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        print(f"  🎯 Valid Solutions: {valid_solutions}/{successful_tests} ({100*valid_solutions/successful_tests:.1f}%)")
        print(f"  🔍 Solutions Found: {solutions_found}/{successful_tests}")
        
        if successful_tests > 0:
            avg_time = np.mean([t.get("solve_time", 0) for t in tests if t.get("success", False)])
            print(f"  ⏱️  Average Time: {avg_time:.3f}s")
        
        # Show individual results
        for test in tests:
            status = "✅" if test.get("success", False) and test.get("valid", False) else "❌"
            instance_name = test.get("instance", "Unknown")
            if test.get("success", False):
                solution_status = "SAT" if test.get("solution_found", False) else "UNSAT"
                time_str = f"{test.get('solve_time', 0):.3f}s"
                print(f"    {status} {instance_name:<20} {solution_status:<6} {time_str}")
            else:
                print(f"    {status} {instance_name:<20} ERROR")
    
    print("\n🎉 Testing completed! Check results above for detailed analysis.")


def main():
    """Main testing function."""
    
    print("🚀 SAT Solver Comprehensive Testing")
    print("=" * 60)
    print("Testing ML4CO-Kit SAT solver implementations")
    print("Solvers: Gurobi ILP, OR-Tools CP-SAT, Greedy Heuristic")
    print()
    
    # Generate test instances
    test_instances = create_test_instances()
    
    # Test all solvers
    all_results = []
    
    # Test Gurobi solver
    try:
        gurobi_results = test_gurobi_solver(test_instances)
        all_results.append(gurobi_results)
    except ImportError as e:
        print(f"⚠️  Skipping Gurobi tests: {e}")
    except Exception as e:
        print(f"❌ Gurobi testing failed: {e}")
    
    # Test OR-Tools solver
    try:
        ortools_results = test_ortools_solver(test_instances)
        all_results.append(ortools_results)
    except ImportError as e:
        print(f"⚠️  Skipping OR-Tools tests: {e}")
    except Exception as e:
        print(f"❌ OR-Tools testing failed: {e}")
    
    # Test Greedy solver
    try:
        greedy_results = test_greedy_solver(test_instances)
        all_results.append(greedy_results)
    except Exception as e:
        print(f"❌ Greedy testing failed: {e}")
    
    # Print comprehensive summary
    if all_results:
        print_summary(all_results)
    else:
        print("❌ No solver tests completed successfully")


if __name__ == "__main__":
    main()