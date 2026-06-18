r"""
Base classes for all combinatorial optimization tasks in ML4CO-Kit.

A **task** represents a single problem instance: input data, an optional candidate
solution (``sol``), and an optional reference solution (``ref_sol``). Subclasses
implement problem-specific encodings, constraint checking, and objective evaluation.
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
from ml4co_kit.utils.pickle_utils import load_pickle
from ml4co_kit.utils.file_utils import check_file_path


class TASK_TYPE(str, Enum):
    """Define the task types as an enumeration."""

    # 1. Routing Problems (Routing)

    # 1.1 TSP Variants
    TSP = "TSP" # Traveling Salesman Problem
    ATSP = "ATSP" # Asymmetric Traveling Salesman Problem 
    OP = "OP" # Orienteering Problem 
    PCTSP = "PCTSP" # Prize Collection Traveling Salesman Problem
    SPCTSP = "SPCTSP" # Stochastic Prize Collection Traveling Salesman Problem
    
    # 1.2 VRP Variants
    CVRP = "CVRP" # Capacitated Vehicle Routing Problem
    CVRPB = "CVRPB" # B/MB: Backhauls and Mixed Backhauls
    CVRPL = "CVRPL" # L: Route Length Limit
    CVRPTW = "CVRPTW" # TW: Time Windows
    CVRPBL = "CVRPBL" # B/MB and L
    CVRPBTW = "CVRPBTW" # B/MB and TW
    CVRPLTW = "CVRPLTW" # L and TW
    CVRPBLTW = "CVRPBLTW" # B/MB and L and TW
    MTVRP = "MTVRP" # MTV: Multi-Task VRP (B/MB, O, TW, L) 

    # 2. Graph Problems (Graph)
    MCL = "MCl" # Maximum Clique
    MCUT = "MCut" # Maximum Cut
    MIS = "MIS" # Maximum Independent Set
    MVC = "MVC" # Minimum Vertex Cover
    
    # 3. Quadratic Assignment Problems (QAP)
    GM = "GM" # Graph Matching
    GED = "GED" # Graph Edit Distance
    KQAP = "KQAP" # Koopmans-Beckmann QAP
    LQAP = "LQAP" # Lawler QAP
    
    # 4. Mixed Integer Programming Problems (MIP)
    MIP = "MIP" # Mixed Integer Programming
    MILP = "MILP" # Mixed Integer Linear Programming
    LP = "LP" # Linear Program

    # 5. Portfolio Optimization Problems (Portfolio)
    MAXRETPO = "MaxRetPO" # Maximum Return Portfolio Optimization
    MINVARPO = "MinVarPO" # Minimum Variance Portfolio Optimization
    MOPO = "MOPO" # Multi-Objective Portfolio Optimization

    # 6. Boolean Satisfiability Problems (SAT)
    SATP = "SAT-P" # Satisfiability Prediction Problem
    SATA = "SAT-A" # Satisfying Assignment Prediction

    # 7. Electronic Design Automation Problems (EDA)
    EDAP = "EDA-P" # EDA Placement
    EDATDP = "EDA-TDP" # EDA Timing-Driven Placement
    EDAR = "EDA-R" # EDA Routing


class TaskBase(object):
    """Base class for a single combinatorial optimization instance.

    Parameters
    ----------
    task_type : TASK_TYPE
        Problem identifier.
    minimize : bool
        Whether the objective is minimized (``True``) or maximized (``False``).
    precision : np.float32 or np.float64, optional
        Floating-point dtype for coordinates and costs. Default is ``np.float32``.

    Attributes
    ----------
    sol : np.ndarray or None
        Current solution encoding (subclass-specific).
    ref_sol : np.ndarray or None
        Reference solution for benchmarking.
    name : str
        Unique instance name (UUID hex by default).
    cache : dict
        Optional cache used by solvers / optimizers.

    Examples
    --------
    >>> import pathlib
    >>> from ml4co_kit import CVRPTask
    >>> task = CVRPTask()
    >>> task.from_pickle(
    ...     pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl")
    ... )
    >>> task.evaluate(task.ref_sol)
    10.973...
    """

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
        """Restore a task from a pickle file.

        Parameters
        ----------
        file_path : pathlib.Path
            Path to a ``.pkl`` file produced by :meth:`to_pickle` or the toolkit.
        """
        with open(file_path, "rb") as file:
            loaded_instance: TaskBase = load_pickle(file)
        self.__dict__.update(loaded_instance.__dict__)
        self._repair_pickle_state()
    
    def _repair_pickle_state(self):
        """Fill missing attributes after loading legacy pickle files."""
        fresh = self.__class__()
        for key, value in fresh.__dict__.items():
            if key not in self.__dict__:
                self.__dict__[key] = value
    
    def _restore_raw_data(self):
        """Restore the original data from cache."""
        pass
    
    def to_pickle(self, file_path: pathlib.Path):
        """Serialize this task to a pickle file.

        Parameters
        ----------
        file_path : pathlib.Path
            Output ``.pkl`` path (parent directories are created if needed).
        """
        check_file_path(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
            f.close()
    
    def from_data(self):
        """Create a problem instance from raw data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the given solution satisfies all problem constraints. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def evaluate(self, sol: np.ndarray, check_constr: bool = True) -> np.floating:
        """Evaluate the given solution. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate_w_gap(self, check_constr: bool = True) -> Sequence[np.floating]:
        """Compare ``sol`` and ``ref_sol`` and return the optimality gap (%).

        Parameters
        ----------
        check_constr : bool, optional
            If ``True``, validate solutions before evaluation.

        Returns
        -------
        tuple of (sol_cost, ref_cost, gap)
            ``gap`` is ``None`` when ``ref_cost`` is near zero. For minimization,
            ``gap = (sol_cost - ref_cost) / ref_cost * 100``.
        """
        # Check if the solution and reference solution are not None
        if self.sol is None or self.ref_sol is None:
            raise ValueError("Solution and reference solution cannot be None!")
        
        # Evaluate the solution and reference solution
        sol_cost = self.evaluate(self.sol, check_constr=check_constr)
        ref_cost = self.evaluate(self.ref_sol, check_constr=check_constr)

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
        ignore_list = ['dist_eval', 'name', 'g1', 'g2', 'affn_builder']
        
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
