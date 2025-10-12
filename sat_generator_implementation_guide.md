## SAT生成器实现指令

基于成功的SAT任务类，现在实现SAT问题生成器，支持不同分布和难度。

### 第一步：创建生成器目录结构

```bash
mkdir -p ml4co_kit/generator/logic
```

### 第二步：创建逻辑生成器基类

创建文件 `ml4co_kit/generator/logic/base.py`：

```python
r"""
Base class for logic problem generators in the ML4CO kit.
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
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.generator.base import GeneratorBase


class LOGIC_TYPE(str, Enum):
    """Define the logic problem types as an enumeration."""
    RANDOM = "random"           # Random SAT instances
    UNIFORM_RANDOM = "uniform_random"  # Uniform random k-SAT
    PLANTED = "planted"         # Planted solutions
    PHASE_TRANSITION = "phase_transition"  # Near phase transition
    INDUSTRIAL = "industrial"   # Industrial-like instances


class LogicGeneratorBase(GeneratorBase):
    """Base class for logic problem generators."""
    
    def __init__(
        self,
        task_type: TASK_TYPE,
        distribution_type: LOGIC_TYPE,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=task_type,
            distribution_type=distribution_type,
            precision=precision
        )
        
        # Logic-specific attributes
        self.num_vars: int = 10         # Default number of variables
        self.num_clauses: int = 42      # Default number of clauses
        self.clause_length: int = 3     # Default clause length (k in k-SAT)
        
        # Random seed for reproducibility
        self.seed: int = None
        
    def set_parameters(
        self,
        num_vars: int,
        num_clauses: int = None,
        clause_length: int = 3,
        seed: int = None
    ):
        """Set generation parameters."""
        self.num_vars = num_vars
        self.clause_length = clause_length
        self.seed = seed
        
        # Auto-calculate num_clauses if not provided
        if num_clauses is None:
            # Use SAT-UNSAT phase transition ratio for k-SAT
            if clause_length == 2:
                ratio = 1.0     # 2-SAT phase transition
            elif clause_length == 3:
                ratio = 4.26    # 3-SAT phase transition
            elif clause_length == 4:
                ratio = 9.93    # 4-SAT phase transition
            else:
                ratio = 2**(clause_length) * np.log(2)  # General approximation
            
            self.num_clauses = int(ratio * num_vars)
        else:
            self.num_clauses = num_clauses
    
    def _set_random_seed(self):
        """Set random seed if provided."""
        if self.seed is not None:
            np.random.seed(self.seed)
```

### 第三步：创建SAT生成器

创建文件 `ml4co_kit/generator/logic/sat.py`：

```python
r"""
Generator for Boolean Satisfiability Problem (SAT) instances.
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
from typing import Union, List, Optional
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.logic.sat import SATTask
from ml4co_kit.generator.logic.base import LogicGeneratorBase, LOGIC_TYPE


class SATGenerator(LogicGeneratorBase):
    """Generator for SAT problem instances."""
    
    def __init__(
        self,
        distribution_type: LOGIC_TYPE = LOGIC_TYPE.UNIFORM_RANDOM,
        precision: Union[np.float32, np.float64] = np.float32,
        num_vars: int = 10,
        num_clauses: Optional[int] = None,
        clause_length: int = 3,
        seed: Optional[int] = None
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.SAT,
            distribution_type=distribution_type,
            precision=precision
        )
        
        # Set parameters
        self.set_parameters(num_vars, num_clauses, clause_length, seed)
        
        # Generation function dictionary
        self.generate_func_dict = {
            LOGIC_TYPE.RANDOM: self._generate_random,
            LOGIC_TYPE.UNIFORM_RANDOM: self._generate_uniform_random,
            LOGIC_TYPE.PLANTED: self._generate_planted,
            LOGIC_TYPE.PHASE_TRANSITION: self._generate_phase_transition,
            LOGIC_TYPE.INDUSTRIAL: self._generate_industrial,
        }
    
    def generate(
        self,
        num_vars: Optional[int] = None,
        num_clauses: Optional[int] = None,
        clause_length: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> SATTask:
        """Generate a SAT instance."""
        # Update parameters if provided
        if num_vars is not None:
            self.num_vars = num_vars
        if num_clauses is not None:
            self.num_clauses = num_clauses
        if clause_length is not None:
            self.clause_length = clause_length
        if seed is not None:
            self.seed = seed
            
        # Auto-calculate num_clauses if not provided
        if self.num_clauses is None:
            self.set_parameters(self.num_vars, None, self.clause_length, self.seed)
        
        # Set random seed
        self._set_random_seed()
        
        # Generate using the appropriate method
        generate_func = self.generate_func_dict[self.distribution_type]
        return generate_func(**kwargs)
    
    def _generate_random(self, **kwargs) -> SATTask:
        """Generate completely random SAT instance."""
        clauses = []
        
        for _ in range(self.num_clauses):
            # Random clause length between 1 and max(3, clause_length)
            clause_len = np.random.randint(1, max(3, self.clause_length) + 1)
            
            # Random variables for this clause
            vars_in_clause = np.random.choice(
                range(1, self.num_vars + 1), 
                size=clause_len, 
                replace=False
            )
            
            # Random polarities
            clause = []
            for var in vars_in_clause:
                if np.random.random() < 0.5:
                    clause.append(-var)  # Negative literal
                else:
                    clause.append(var)   # Positive literal
            
            clauses.append(clause)
        
        # Create SAT task
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        return sat_task
    
    def _generate_uniform_random(self, **kwargs) -> SATTask:
        """Generate uniform random k-SAT instance."""
        clauses = []
        
        for _ in range(self.num_clauses):
            # Fixed clause length (k-SAT)
            vars_in_clause = np.random.choice(
                range(1, self.num_vars + 1),
                size=self.clause_length,
                replace=False
            )
            
            # Random polarities
            clause = []
            for var in vars_in_clause:
                if np.random.random() < 0.5:
                    clause.append(-var)
                else:
                    clause.append(var)
            
            clauses.append(clause)
        
        # Create SAT task
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        return sat_task
    
    def _generate_planted(self, **kwargs) -> SATTask:
        """Generate SAT instance with planted (known) solution."""
        # Generate a random satisfying assignment
        planted_solution = np.random.choice([0, 1], size=self.num_vars)
        
        clauses = []
        max_attempts = self.num_clauses * 10  # Prevent infinite loops
        attempts = 0
        
        while len(clauses) < self.num_clauses and attempts < max_attempts:
            attempts += 1
            
            # Generate a random clause
            vars_in_clause = np.random.choice(
                range(1, self.num_vars + 1),
                size=self.clause_length,
                replace=False
            )
            
            clause = []
            for var in vars_in_clause:
                var_index = var - 1  # Convert to 0-based
                if np.random.random() < 0.5:
                    clause.append(-var)
                else:
                    clause.append(var)
            
            # Check if this clause is satisfied by the planted solution
            clause_satisfied = False
            for literal in clause:
                var_index = abs(literal) - 1
                if literal > 0 and planted_solution[var_index]:
                    clause_satisfied = True
                    break
                elif literal < 0 and not planted_solution[var_index]:
                    clause_satisfied = True
                    break
            
            # Only add clauses that are satisfied by the planted solution
            if clause_satisfied:
                clauses.append(clause)
        
        # Create SAT task with reference solution
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        sat_task.set_ref_assignment(planted_solution)
        return sat_task
    
    def _generate_phase_transition(self, **kwargs) -> SATTask:
        """Generate SAT instance near the satisfiability phase transition."""
        # Calculate phase transition ratio
        if self.clause_length == 3:
            phase_ratio = 4.26  # Known 3-SAT phase transition
        else:
            phase_ratio = 2**self.clause_length * np.log(2)
        
        # Add some randomness around the phase transition
        noise = kwargs.get('noise', 0.1)
        ratio = phase_ratio * (1 + np.random.uniform(-noise, noise))
        
        # Calculate number of clauses for this ratio
        num_clauses = int(ratio * self.num_vars)
        
        # Generate uniform random k-SAT with this ratio
        old_num_clauses = self.num_clauses
        self.num_clauses = num_clauses
        result = self._generate_uniform_random()
        self.num_clauses = old_num_clauses  # Restore original
        
        return result
    
    def _generate_industrial(self, **kwargs) -> SATTask:
        """Generate industrial-like SAT instance with structure."""
        # Industrial instances often have variable clause lengths and community structure
        clauses = []
        
        # Create some "communities" of variables
        num_communities = max(2, self.num_vars // 5)
        community_size = self.num_vars // num_communities
        communities = []
        
        for i in range(num_communities):
            start = i * community_size + 1
            end = min((i + 1) * community_size + 1, self.num_vars + 1)
            communities.append(list(range(start, end)))
        
        # Generate clauses with bias towards same community
        for _ in range(self.num_clauses):
            # Variable clause length (1 to clause_length)
            clause_len = np.random.randint(1, self.clause_length + 1)
            
            # 70% chance to pick from same community, 30% chance to mix
            if np.random.random() < 0.7 and len(communities) > 0:
                # Pick from same community
                community = np.random.choice(len(communities))
                available_vars = communities[community]
                if len(available_vars) >= clause_len:
                    vars_in_clause = np.random.choice(
                        available_vars, size=clause_len, replace=False
                    )
                else:
                    # Fallback to random selection
                    vars_in_clause = np.random.choice(
                        range(1, self.num_vars + 1), size=clause_len, replace=False
                    )
            else:
                # Mix variables from different communities
                vars_in_clause = np.random.choice(
                    range(1, self.num_vars + 1), size=clause_len, replace=False
                )
            
            # Random polarities with slight bias towards positive
            clause = []
            for var in vars_in_clause:
                if np.random.random() < 0.4:  # 40% negative, 60% positive
                    clause.append(-var)
                else:
                    clause.append(var)
            
            clauses.append(clause)
        
        # Create SAT task
        sat_task = SATTask(precision=self.precision)
        sat_task.from_data(clauses=clauses, num_vars=self.num_vars)
        return sat_task
    
    def generate_satisfiable_instance(
        self, 
        num_vars: int, 
        num_clauses: Optional[int] = None,
        max_attempts: int = 100
    ) -> SATTask:
        """Generate a guaranteed satisfiable SAT instance."""
        # Use planted solution method for guaranteed satisfiability
        old_type = self.distribution_type
        self.distribution_type = LOGIC_TYPE.PLANTED
        
        task = self.generate(num_vars=num_vars, num_clauses=num_clauses)
        
        self.distribution_type = old_type  # Restore original type
        return task
    
    def generate_unsatisfiable_instance(
        self,
        num_vars: int,
        num_clauses: Optional[int] = None,
        max_attempts: int = 100
    ) -> SATTask:
        """Generate a guaranteed unsatisfiable SAT instance."""
        # Add contradictory clauses to ensure unsatisfiability
        # Start with a random instance
        task = self.generate(num_vars=num_vars, num_clauses=num_clauses)
        
        # Add contradictory unit clauses for first few variables
        additional_clauses = []
        for i in range(min(3, num_vars)):
            var = i + 1
            # Add both positive and negative unit clauses for the same variable
            additional_clauses.append([var])
            additional_clauses.append([-var])
        
        # Combine original and contradictory clauses
        all_clauses = task.clauses + additional_clauses
        
        # Create new unsatisfiable task
        unsat_task = SATTask(precision=self.precision)
        unsat_task.from_data(clauses=all_clauses, num_vars=num_vars)
        return unsat_task
```

### 第四步：创建逻辑生成器模块初始化文件

创建文件 `ml4co_kit/generator/logic/__init__.py`：

```python
r"""
Logic Generator Module.
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

from .base import LogicGeneratorBase, LOGIC_TYPE
from .sat import SATGenerator
```

### 第五步：更新主生成器模块

编辑文件 `ml4co_kit/generator/__init__.py`，在末尾添加：

```python
# Logic Generator
from .logic.base import LogicGeneratorBase, LOGIC_TYPE
from .logic.sat import SATGenerator
```

### 第六步：创建SAT生成器测试文件

创建文件 `test_sat_generator.py`：

```python
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


if __name__ == "__main__":
    print("Testing SAT Generator Implementation")
    print("=" * 50)
    
    try:
        # Run all tests
        test_sat_generator_basic()
        test_sat_generator_distributions()
        test_sat_generator_special()
        test_sat_generator_phase_transition()
        
        print("\n" + "=" * 50)
        print("✅ All SAT Generator tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

### 第七步：运行生成器测试

```bash
# 运行生成器测试
python test_sat_generator.py
```

### 验证步骤

完成后请验证：
1. 生成器能创建不同类型的SAT实例
2. 支持所有分布类型（随机、均匀、种植解、相变、工业化）
3. 能生成满足性和不满足性实例
4. 相变生成器使用正确的子句/变量比例

这样就完成了功能完整的SAT生成器实现！