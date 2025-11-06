r"""
Base class for SAT problems in the ML4CO kit.
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
from typing import Union, List
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path


class SATTaskBase(TaskBase):
    """Base class for SAT problems."""
    
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(SATTaskBase, self).__init__(
            task_type=task_type, 
            minimize=minimize, 
            precision=precision
        )
        
        # Initialize Attributes
        self.vars_num: int = 0                    # Number of variables
        self.clauses: list = []                   # List of clauses
        self.clauses_num: int = 0                 # Number of clauses
        self.satisfiable: bool = None             # Whether the formula is satisfiable

    def _check_sol_dim(self):
        """Ensure solution is a 1D array."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 1D array."""
        raise NotImplementedError("Subclasses should implement this method.")

    def from_data(
        self, 
        clauses: List[List[int]],
        vars_num: int = None,
        sol: Union[np.ndarray, bool] = None,
        ref: bool = False,
        name: str = None
    ):
        # Set Attributes and Check Dimensions
        if clauses is not None:
            self.clauses = clauses
            self.clauses_num = len(clauses)
            if vars_num is None:
                max_var = 0
                for clause in clauses:
                    for literal in clause:
                        max_var = max(max_var, abs(literal))
                vars_num = max_var
            self.vars_num = vars_num
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()
        
        # Set Name if Provided
        if name is not None:
            self.name = name

    def from_cnf(self, file_path: pathlib.Path):
        """Load SAT instance from CNF file."""
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('c'):  # Comment line
                    continue
                elif line.startswith('p cnf'):  # Header line
                    parts = line.split()
                    vars_num = int(parts[2])
                else:  # Clause line
                    clause = [int(x) for x in line.split() if int(x) != 0]
                    if clause:  # Only add non-empty clauses
                        self.clauses.append(clause)
        self.from_data(clauses=self.clauses, vars_num=vars_num)

    def to_cnf(self, file_path: pathlib.Path):
        """Save SAT instance to CNF file."""
        check_file_path(file_path)
        with open(file_path, 'w') as f:
            f.write(f"p cnf {self.vars_num} {self.clauses_num}\n")
            for clause in self.clauses:
                clause_str = ' '.join(map(str, clause)) + ' 0\n'
                f.write(clause_str)