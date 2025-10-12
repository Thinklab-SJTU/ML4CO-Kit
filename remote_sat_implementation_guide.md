## 远程服务器SAT任务类实现指令

请按照以下步骤在ML4CO-Kit项目中实现SAT (Boolean Satisfiability Problem) 任务类：

### 第一步：修改基础任务类型定义

1. 编辑文件 `ml4co_kit/task/base.py`
2. 在 `TASK_TYPE` 枚举中添加SAT类型
3. 在第45行左右（LP定义之后）添加：

```python
    # Linear Programming Problems
    LP = "LP" # Linear Program

    # Logic Problems
    SAT = "SAT" # Boolean Satisfiability Problem
```

### 第二步：创建逻辑任务模块目录结构

```bash
mkdir -p ml4co_kit/task/logic
```

### 第三步：创建逻辑任务基类

创建文件 `ml4co_kit/task/logic/base.py`：

```python
r"""
Base class for logic problems in the ML4CO kit.
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
from typing import Union
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class LogicTaskBase(TaskBase):
    """Base class for logic problems."""
  
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool = False,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=task_type, 
            minimize=minimize, 
            precision=precision
        )
      
        # Logic-specific attributes
        self.num_vars: int = 0                    # Number of variables
        self.clauses: list = []                   # List of clauses
        self.assignment: np.ndarray = None        # Current variable assignment
        self.ref_assignment: np.ndarray = None    # Reference assignment
  
    def get_num_vars(self) -> int:
        """Get the number of variables."""
        return self.num_vars
  
    def get_num_clauses(self) -> int:
        """Get the number of clauses."""
        return len(self.clauses)
  
    def set_assignment(self, assignment: np.ndarray):
        """Set the variable assignment."""
        if len(assignment) != self.num_vars:
            raise ValueError(f"Assignment length {len(assignment)} != num_vars {self.num_vars}")
        self.assignment = assignment.astype(np.bool_)
        # Also set sol for compatibility with base class
        self.sol = assignment.astype(np.int32)
  
    def set_ref_assignment(self, ref_assignment: np.ndarray):
        """Set the reference variable assignment."""
        if len(ref_assignment) != self.num_vars:
            raise ValueError(f"Reference assignment length {len(ref_assignment)} != num_vars {self.num_vars}")
        self.ref_assignment = ref_assignment.astype(np.bool_)
        # Also set ref_sol for compatibility with base class
        self.ref_sol = ref_assignment.astype(np.int32)
```

### 第四步：创建SAT任务类

创建文件 `ml4co_kit/task/logic/sat.py`：

```python
r"""
Boolean Satisfiability Problem (SAT).

SAT is the problem of determining if there exists an interpretation 
that satisfies a given Boolean formula. It was the first problem 
to be proven NP-complete.
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


import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.logic.base import LogicTaskBase


class SATTask(LogicTaskBase):
    """Boolean Satisfiability Problem (SAT) task."""
  
    def __init__(
        self, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.SAT, 
            minimize=False,  # SAT is a decision problem, but we maximize satisfied clauses
            precision=precision
        )
      
        # SAT-specific attributes
        self.satisfied_clauses: Optional[np.ndarray] = None  # Which clauses are satisfied
      
    def from_data(
        self, 
        clauses: List[List[int]], 
        num_vars: Optional[int] = None,
        assignment: Optional[np.ndarray] = None,
        ref_assignment: Optional[np.ndarray] = None
    ):
        """Create SAT instance from clause data.
      
        Args:
            clauses: List of clauses, each clause is a list of literals
                    Positive literals: variable is True
                    Negative literals: variable is False
                    Example: [[1, -2, 3], [-1, 2], [2, -3]] means:
                    (x1 OR NOT x2 OR x3) AND (NOT x1 OR x2) AND (x2 OR NOT x3)
            num_vars: Number of variables (auto-detected if None)
            assignment: Current variable assignment
            ref_assignment: Reference assignment (if known)
        """
        self.clauses = clauses
      
        # Auto-detect number of variables if not provided
        if num_vars is None:
            max_var = 0
            for clause in clauses:
                for literal in clause:
                    max_var = max(max_var, abs(literal))
            self.num_vars = max_var
        else:
            self.num_vars = num_vars
          
        # Set assignments if provided
        if assignment is not None:
            self.set_assignment(assignment)
        if ref_assignment is not None:
            self.set_ref_assignment(ref_assignment)
  
    def from_dimacs(self, file_path: pathlib.Path):
        """Load SAT instance from DIMACS CNF format.
      
        DIMACS format:
        - Lines starting with 'c' are comments
        - Line starting with 'p cnf <num_vars> <num_clauses>' is the header
        - Each following line is a clause ending with 0
        """
        clauses = []
        num_vars = 0
      
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('c'):  # Comment line
                    continue
                elif line.startswith('p cnf'):  # Header line
                    parts = line.split()
                    num_vars = int(parts[2])
                    # num_clauses = int(parts[3])  # Not needed for parsing
                else:  # Clause line
                    clause = [int(x) for x in line.split() if int(x) != 0]
                    if clause:  # Only add non-empty clauses
                        clauses.append(clause)
      
        self.from_data(clauses=clauses, num_vars=num_vars)
  
    def to_dimacs(self, file_path: pathlib.Path):
        """Save SAT instance to DIMACS CNF format."""
        check_file_path(file_path)
      
        with open(file_path, 'w') as f:
            # Write header
            f.write(f"p cnf {self.num_vars} {len(self.clauses)}\n")
          
            # Write clauses
            for clause in self.clauses:
                clause_str = ' '.join(map(str, clause)) + ' 0\n'
                f.write(clause_str)
  
    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the given assignment is valid.
      
        For SAT, any boolean assignment is valid in terms of format,
        but we check if it satisfies all clauses.
        """
        if len(sol) != self.num_vars:
            return False
      
        # Check if all values are 0 or 1
        return np.all((sol == 0) | (sol == 1))
  
    def is_satisfiable(self, assignment: Optional[np.ndarray] = None) -> bool:
        """Check if the given assignment satisfies all clauses."""
        if assignment is None:
            if self.assignment is None:
                raise ValueError("No assignment provided!")
            assignment = self.assignment
      
        return self.evaluate(assignment) == len(self.clauses)
  
    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the assignment by counting satisfied clauses.
      
        Returns the number of satisfied clauses.
        For a satisfiable instance, this should equal the total number of clauses.
        """
        if not self.check_constraints(sol):
            raise ValueError("Invalid assignment!")
      
        assignment = sol.astype(np.bool_)
        satisfied_count = 0
        satisfied_clauses = []
      
        for i, clause in enumerate(self.clauses):
            clause_satisfied = False
            for literal in clause:
                var_index = abs(literal) - 1  # Convert to 0-based indexing
                if var_index >= len(assignment):
                    raise ValueError(f"Variable {abs(literal)} not in assignment!")
              
                # Check if literal is satisfied
                if literal > 0:  # Positive literal
                    if assignment[var_index]:
                        clause_satisfied = True
                        break
                else:  # Negative literal
                    if not assignment[var_index]:
                        clause_satisfied = True
                        break
          
            if clause_satisfied:
                satisfied_count += 1
                satisfied_clauses.append(i)
      
        # Store which clauses are satisfied for analysis
        self.satisfied_clauses = np.array(satisfied_clauses)
      
        return np.array(satisfied_count).astype(self.precision)
  
    def get_unsatisfied_clauses(self, assignment: Optional[np.ndarray] = None) -> List[int]:
        """Get indices of unsatisfied clauses."""
        if assignment is None:
            assignment = self.assignment
      
        if assignment is None:
            raise ValueError("No assignment provided!")
      
        # Evaluate to update satisfied_clauses
        self.evaluate(assignment)
      
        all_clauses = set(range(len(self.clauses)))
        satisfied_set = set(self.satisfied_clauses) if self.satisfied_clauses is not None else set()
      
        return list(all_clauses - satisfied_set)
  
    def get_clause_satisfaction_ratio(self, assignment: Optional[np.ndarray] = None) -> float:
        """Get the ratio of satisfied clauses."""
        if assignment is None:
            assignment = self.assignment
      
        if assignment is None:
            raise ValueError("No assignment provided!")
      
        satisfied_count = self.evaluate(assignment)
        return float(satisfied_count) / len(self.clauses)
  
    def copy(self):
        """Create a copy of the SAT task."""
        new_task = SATTask(precision=self.precision)
        new_task.from_data(
            clauses=self.clauses.copy(),
            num_vars=self.num_vars
        )
        if self.assignment is not None:
            new_task.set_assignment(self.assignment.copy())
        if self.ref_assignment is not None:
            new_task.set_ref_assignment(self.ref_assignment.copy())
        return new_task
  
    def render(
        self, 
        save_path: pathlib.Path,
        with_assignment: bool = True,
        figsize: tuple = (12, 8),
        satisfied_color: str = "green",
        unsatisfied_color: str = "red",
        var_true_color: str = "lightblue",
        var_false_color: str = "lightcoral"
    ):
        """Render the SAT problem instance and assignment."""
        check_file_path(save_path)
      
        if with_assignment and self.assignment is None:
            raise ValueError("Assignment is not provided!")
      
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
      
        # Plot 1: Variable assignment
        if with_assignment:
            ax1.set_title("Variable Assignment")
            var_colors = [var_true_color if val else var_false_color 
                         for val in self.assignment]
            bars1 = ax1.bar(range(1, self.num_vars + 1), 
                           [1] * self.num_vars, color=var_colors)
            ax1.set_xlabel("Variable")
            ax1.set_ylabel("Value")
            ax1.set_ylim(0, 1.2)
          
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars1, self.assignment)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(val)}', ha='center', va='bottom')
        else:
            ax1.set_title("SAT Instance")
            ax1.text(0.5, 0.5, f"Variables: {self.num_vars}\nClauses: {len(self.clauses)}", 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.axis('off')
      
        # Plot 2: Clause satisfaction
        if with_assignment:
            satisfied_count = int(self.evaluate(self.assignment))
            unsatisfied_count = len(self.clauses) - satisfied_count
          
            ax2.set_title("Clause Satisfaction")
            labels = ['Satisfied', 'Unsatisfied']
            sizes = [satisfied_count, unsatisfied_count]
            colors = [satisfied_color, unsatisfied_color]
          
            if unsatisfied_count == 0:
                ax2.pie([1], labels=['All Satisfied'], colors=[satisfied_color], autopct='%1.1f%%')
                ax2.text(0, -1.3, f"✓ SATISFIABLE", ha='center', va='center', 
                        color=satisfied_color, fontsize=14, weight='bold')
            else:
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                ax2.text(0, -1.3, f"✗ {unsatisfied_count} unsatisfied", ha='center', va='center', 
                        color=unsatisfied_color, fontsize=12)
        else:
            ax2.set_title("Clause Statistics")
            clause_lengths = [len(clause) for clause in self.clauses]
            ax2.hist(clause_lengths, bins=range(1, max(clause_lengths) + 2), 
                    alpha=0.7, edgecolor='black')
            ax2.set_xlabel("Clause Length")
            ax2.set_ylabel("Number of Clauses")
      
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
  
    def __repr__(self):
        return f"SATTask(vars={self.num_vars}, clauses={len(self.clauses)})"
```

### 第五步：创建逻辑任务模块初始化文件

创建文件 `ml4co_kit/task/logic/__init__.py`：

```python
r"""
Logic Task Module.
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

from .base import LogicTaskBase
from .sat import SATTask
```

### 第六步：更新主任务模块

编辑文件 `ml4co_kit/task/__init__.py`，在文件末尾添加：

```python
# Logic Task
from .logic.base import LogicTaskBase
from .logic.sat import SATTask
```

### 第七步：创建测试文件

创建文件 `test_sat_task.py`：

```python
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


if __name__ == "__main__":
    print("Testing SAT Task Implementation")
    print("=" * 50)
  
    try:
        # Run all tests
        test_sat_task_basic()
        test_sat_task_evaluation()
      
        print("\n" + "=" * 50)
        print("✅ All SAT Task tests passed!")
      
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

### 第八步：运行测试

```bash
# 激活环境
source /mnt/nas-new/home/zhanghang/zhangxihe/ml4co_venv/bin/activate

# 运行测试
python test_sat_task.py
```

### 验证步骤

完成后请验证：

1. 所有文件创建无误
2. 测试脚本能正常运行
3. 输出显示SAT任务功能正常

这样就完成了SAT任务类的完整实现，为后续的Generator、Solver、Optimizer和Wrapper奠定了基础。
