r"""
Base class for all problems in the ML4CO kit.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import uuid
import pickle
import pathlib
import hashlib
import numpy as np
from enum import Enum
from typing import Sequence, Union
from ml4co_kit.utils.file_utils import check_file_path


class TASK_TYPE(str, Enum):
    """Define the task types as an enumeration."""

    # Routing Problems
    ATSP = "ATSP" # Asymmetric Traveling Salesman Problem 
    CVRP = "CVRP" # Capacitated Vehicle Routing Problem
    OP = "OP" # Orienteering Problem 
    PCTSP = "PCTSP" # Prize Collection Traveling Salesman Problem
    SPCTSP = "SPCTSP" # Stochastic Prize Collection Traveling Salesman Problem
    TSP = "TSP" # Traveling Salesman Problem

    # Graph Problems
    MCL = "MCl" # Maximum Clique
    MCUT = "MCut" # Maximum Cut
    MIS = "MIS" # Maximum Independent Set
    MVC = "MVC" # Minimum Vertex Cover

    # Knapsack Problems
    KP = "KP" # Knapsack Problem

    # Linear Programming Problems
    LP = "LP" # Linear Program

    # Portfolio Optimization Problems
    MAXRETPO = "MaxRetPO" # Maximum Return Portfolio Optimization
    MINVARPO = "MinVarPO" # Minimum Variance Portfolio Optimization
    MOPO = "MOPO" # Multi-Objective Portfolio Optimization


class TaskBase(object):
    """Base class for all tasks in the ML4CO kit."""

    def __init__(
        self, 
        task_type: TASK_TYPE, 
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        self.task_type = task_type          # Task type
        self.minimize = minimize            # Whether to minimize the objective function
        self.precision = precision          # Precision
        self.sol: np.ndarray = None         # Solution
        self.ref_sol: np.ndarray = None     # Reference solution
        self.cache: dict = {}               # Cache (used for optimization)
        self.name: str = uuid.uuid4().hex   # Name of the instance
    
    def _check_sol_not_none(self):
        """Check if solution is not None."""
        if self.sol is None:
            raise ValueError("``sol`` cannot be None!")

    def _check_ref_sol_not_none(self):
        """Check if reference solution is not None."""
        if self.ref_sol is None:
            raise ValueError("``ref_sol`` cannot be None!")
    
    def from_pickle(self, file_path: pathlib.Path):
        """Create a problem instance from a pickle file."""
        with open(file_path, "rb") as file:
            loaded_instance: TaskBase = pickle.load(file)
        self.__dict__.update(loaded_instance.__dict__)
    
    def to_pickle(self, file_path: pathlib.Path):
        # Check file path
        check_file_path(file_path)
        
        # Save task data to ``.pkl`` file
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
            f.close()
    
    def from_data(self):
        """Create a problem instance from raw data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the given solution satisfies all problem constraints. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the given solution. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate_w_gap(self) -> Sequence[np.floating]:
        """Evaluate the given solution with gap."""
        # Check if the solution and reference solution are not None
        if self.sol is None or self.ref_sol is None:
            raise ValueError("Solution and reference solution cannot be None!")
        
        # Evaluate the solution and reference solution
        sol_cost = self.evaluate(self.sol)
        ref_cost = self.evaluate(self.ref_sol)

        # Calculate the gap
        if abs(ref_cost) < 1e-8:
            gap = None
        else:
            if self.minimize:
                gap = (sol_cost - ref_cost) / ref_cost
            else:
                gap = (ref_cost - sol_cost) / ref_cost
            gap = gap * np.array(100.0).astype(self.precision)
        
        return sol_cost, ref_cost, gap
    
    def render(self):
        """Render the problem instance. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_data_md5(self) -> str:
        """
        Calculate MD5 hash of the task's data content.
        
        This method computes the MD5 hash based on the actual data content
        rather than the file content, which is useful for verifying data
        integrity when pickle files may have different object references.
        
        Returns:
            str: MD5 hash of the task's data content
        """
        data_parts = []
        ignore_list = ['dist_eval', 'name']
        
        # Get all attributes from __dict__ except dist_eval (which contains object references)
        task_dict = {k: v for k, v in self.__dict__.items() if k not in ignore_list}
        
        # Sort keys for consistent ordering
        for key in sorted(task_dict.keys()):
            value = task_dict[key]
            
            # Handle numpy arrays
            if isinstance(value, np.ndarray) and value is not None:
                data_parts.append(value.tobytes())
            # Handle other data types
            elif value is not None:
                data_parts.append(str(value).encode())
        
        # Combine all data and compute MD5
        combined_data = b''.join(data_parts)
        return hashlib.md5(combined_data).hexdigest()
    
    def __repr__(self):
        return f"{self.task_type.value}Task({self.name})"