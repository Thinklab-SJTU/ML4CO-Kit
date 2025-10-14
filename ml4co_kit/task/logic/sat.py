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
        
        # Initialize core attributes
        self.clauses: List[List[int]] = []
        self.num_vars: int = 0
    
    @property
    def n_vars(self) -> int:
        """Alias for num_vars to maintain compatibility with solvers."""
        return self.num_vars
    
    @property
    def cnf(self) -> List[List[int]]:
        """Alias for clauses to maintain compatibility with solvers."""
        return self.clauses
    
    @property
    def solution(self) -> Optional[np.ndarray]:
        """Alias for sol to maintain compatibility with test scripts."""
        return self.sol
    
    @solution.setter
    def solution(self, value: Optional[np.ndarray]):
        """Setter for solution alias."""
        self.sol = value
        
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