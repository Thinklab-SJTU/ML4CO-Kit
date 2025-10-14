#!/usr/bin/env python3
"""
Quick SAT Solver Test Script - Simplified Version for Debugging
"""

import os
import sys
import time
import numpy as np

# Add ML4CO-Kit to path
sys.path.append('/mnt/nas-new/home/zhanghang/zhangxihe/ml4co_workspace/ML4CO-Kit')

from ml4co_kit.task.logic.sat import SATTask
from ml4co_kit.solver.gurobi import GurobiSolver

def test_simple_sat():
    """Test basic SAT functionality with a simple example."""
    
    print("🧪 Testing Basic SAT Functionality")
    print("=" * 50)
    
    # Create simple SAT instance: (x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2)
    sat_task = SATTask()
    simple_cnf = [
        [1, 2],     # (x1 ∨ x2)
        [-1, 2],    # (¬x1 ∨ x2)  
        [1, -2]     # (x1 ∨ ¬x2)
    ]
    
    print("Creating SAT instance...")
    sat_task.from_data(clauses=simple_cnf, num_vars=2)
    
    print(f"✅ SAT instance created:")
    print(f"   Variables: {sat_task.n_vars}")
    print(f"   Clauses: {len(sat_task.cnf)}")
    print(f"   CNF: {sat_task.cnf}")
    
    # Test Gurobi solver
    print("\n🔧 Testing Gurobi solver...")
    try:
        solver = GurobiSolver(gurobi_time_limit=10.0)
        start_time = time.time()
        solver.solve(sat_task)
        solve_time = time.time() - start_time
        
        print(f"   ⏱️ Solve time: {solve_time:.3f}s")
        print(f"   📊 Solution found: {sat_task.solution is not None}")
        
        if sat_task.solution is not None:
            print(f"   🎯 Solution: {sat_task.solution}")
            
            # Validate solution
            is_satisfiable = sat_task.is_satisfiable(sat_task.solution)
            print(f"   ✅ Is satisfiable: {is_satisfiable}")
        
        print("✅ Gurobi test passed!")
        
    except Exception as e:
        print(f"❌ Gurobi test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_sat()