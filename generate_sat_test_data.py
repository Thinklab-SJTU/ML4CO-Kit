#!/usr/bin/env python3
"""
Generate SAT test data following ML4CO-Kit data structure patterns.

This script creates test data for SAT problems in both task/ and wrapper/ formats:
- task/: Individual SAT task instances (single .pkl files)
- wrapper/: Collections of SAT instances (.pkl and .txt files)
"""

import os
import sys
import numpy as np
import pickle

# Add ML4CO-Kit to path
sys.path.append(os.path.dirname(__file__))

from ml4co_kit.generator.logic.sat import SATGenerator
from ml4co_kit.task.logic.sat import SATTask


def generate_sat_task_data():
    """Generate individual SAT task data (for task/ directory)."""
    print("🔧 Generating individual SAT task data...")
    
    generator = SATGenerator()
    task_dir = "test_dataset/sat/task"
    os.makedirs(task_dir, exist_ok=True)
    
    # Generate different types of SAT instances
    test_cases = [
        {
            "name": "sat_small_random_task.pkl",
            "params": {"num_vars": 10, "num_clauses": 30, "distribution": "RANDOM", "seed": 42}
        },
        {
            "name": "sat_medium_planted_task.pkl", 
            "params": {"num_vars": 20, "num_clauses": 60, "distribution": "PLANTED", "seed": 123}
        },
        {
            "name": "sat_phase_transition_task.pkl",
            "params": {"num_vars": 15, "num_clauses": 65, "distribution": "PHASE_TRANSITION", "seed": 456}
        },
        {
            "name": "sat_industrial_task.pkl",
            "params": {"num_vars": 25, "num_clauses": 100, "distribution": "INDUSTRIAL", "seed": 789}
        }
    ]
    
    for case in test_cases:
        print(f"  📝 Creating {case['name']}...")
        task = generator.generate(**case["params"])
        task.name = case["name"].replace("_task.pkl", "")
        
        # Save task to pickle
        task_path = os.path.join(task_dir, case["name"])
        task.to_pickle(task_path)
        print(f"    ✅ Saved to {task_path}")
    
    print(f"✅ Generated {len(test_cases)} individual task files")


def generate_sat_wrapper_data():
    """Generate SAT wrapper data (for wrapper/ directory)."""
    print("\n🔧 Generating SAT wrapper data...")
    
    generator = SATGenerator()
    wrapper_dir = "test_dataset/sat/wrapper"
    os.makedirs(wrapper_dir, exist_ok=True)
    
    # Generate collections of SAT instances
    collections = [
        {
            "name": "sat_random_small_4ins",
            "num_instances": 4,
            "params": {"num_vars": 8, "num_clauses": 20, "distribution": "RANDOM"}
        },
        {
            "name": "sat_planted_medium_4ins", 
            "num_instances": 4,
            "params": {"num_vars": 15, "num_clauses": 45, "distribution": "PLANTED"}
        },
        {
            "name": "sat_phase_transition_4ins",
            "num_instances": 4, 
            "params": {"num_vars": 12, "num_clauses": 50, "distribution": "PHASE_TRANSITION"}
        }
    ]
    
    for collection in collections:
        print(f"  📝 Creating {collection['name']} collection...")
        
        # Generate instances
        instances = []
        for i in range(collection["num_instances"]):
            seed = 1000 + i  # Different seed for each instance
            params = collection["params"].copy()
            params["seed"] = seed
            
            task = generator.generate(**params)
            task.name = f"{collection['name']}_instance_{i+1}"
            instances.append(task)
        
        # Save as pickle
        pkl_path = os.path.join(wrapper_dir, f"{collection['name']}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(instances, f)
        print(f"    ✅ Saved pickle to {pkl_path}")
        
        # Save as txt (DIMACS format)
        txt_path = os.path.join(wrapper_dir, f"{collection['name']}.txt")
        with open(txt_path, 'w') as f:
            for i, task in enumerate(instances):
                f.write(f"c Instance {i+1}: {task.name}\n")
                f.write(f"c Variables: {task.num_vars}, Clauses: {task.get_num_clauses()}\n")
                f.write(f"p cnf {task.num_vars} {task.get_num_clauses()}\n")
                
                for clause in task.clauses:
                    clause_str = " ".join(map(str, clause)) + " 0"
                    f.write(clause_str + "\n")
                
                # Add separator between instances
                if i < len(instances) - 1:
                    f.write("c ---\n")
        
        print(f"    ✅ Saved txt to {txt_path}")
    
    print(f"✅ Generated {len(collections)} wrapper collections")


def generate_sat_test_instances():
    """Generate simple test instances for solver testing."""
    print("\n🔧 Generating simple test instances for solver validation...")
    
    wrapper_dir = "test_dataset/sat/wrapper"
    os.makedirs(wrapper_dir, exist_ok=True)
    
    # Create simple, known-satisfiable instances
    simple_instances = []
    
    # Instance 1: Very simple 3-SAT
    task1 = SATTask()
    clauses1 = [
        [1, 2, 3],    # x1 ∨ x2 ∨ x3
        [-1, 2],      # ¬x1 ∨ x2
        [-2, 3],      # ¬x2 ∨ x3
        [1, -3]       # x1 ∨ ¬x3
    ]
    task1.from_data(clauses=clauses1, num_vars=3)
    task1.name = "simple_3sat_instance"
    simple_instances.append(task1)
    
    # Instance 2: Another satisfiable instance
    task2 = SATTask()
    clauses2 = [
        [1, 2],       # x1 ∨ x2
        [-1, 3],      # ¬x1 ∨ x3
        [2, -3],      # x2 ∨ ¬x3
        [-2, 1]       # ¬x2 ∨ x1
    ]
    task2.from_data(clauses=clauses2, num_vars=3)
    task2.name = "simple_satisfiable_instance"
    simple_instances.append(task2)
    
    # Instance 3: Unsatisfiable instance
    task3 = SATTask()
    clauses3 = [
        [1],          # x1
        [-1],         # ¬x1
    ]
    task3.from_data(clauses=clauses3, num_vars=1)
    task3.name = "simple_unsat_instance"
    simple_instances.append(task3)
    
    # Save simple test instances
    pkl_path = os.path.join(wrapper_dir, "sat_simple_test_3ins.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(simple_instances, f)
    print(f"    ✅ Saved simple test instances to {pkl_path}")
    
    # Save as txt
    txt_path = os.path.join(wrapper_dir, "sat_simple_test_3ins.txt")
    with open(txt_path, 'w') as f:
        for i, task in enumerate(simple_instances):
            f.write(f"c Instance {i+1}: {task.name}\n")
            f.write(f"c Variables: {task.num_vars}, Clauses: {task.get_num_clauses()}\n")
            f.write(f"p cnf {task.num_vars} {task.get_num_clauses()}\n")
            
            for clause in task.clauses:
                clause_str = " ".join(map(str, clause)) + " 0"
                f.write(clause_str + "\n")
            
            if i < len(simple_instances) - 1:
                f.write("c ---\n")
    
    print(f"    ✅ Saved simple test instances to {txt_path}")


def main():
    """Main function to generate all SAT test data."""
    print("🚀 Generating SAT Test Data")
    print("=" * 50)
    print("Following ML4CO-Kit data structure patterns...")
    print()
    
    try:
        # Generate individual task data
        generate_sat_task_data()
        
        # Generate wrapper data  
        generate_sat_wrapper_data()
        
        # Generate simple test instances
        generate_sat_test_instances()
        
        print("\n" + "=" * 50)
        print("✅ All SAT test data generated successfully!")
        print("\nGenerated files:")
        print("📁 test_dataset/sat/task/")
        print("   - sat_small_random_task.pkl")
        print("   - sat_medium_planted_task.pkl") 
        print("   - sat_phase_transition_task.pkl")
        print("   - sat_industrial_task.pkl")
        print("📁 test_dataset/sat/wrapper/")
        print("   - sat_random_small_4ins.pkl/.txt")
        print("   - sat_planted_medium_4ins.pkl/.txt")
        print("   - sat_phase_transition_4ins.pkl/.txt")
        print("   - sat_simple_test_3ins.pkl/.txt")
        
    except Exception as e:
        print(f"\n❌ Failed to generate test data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()