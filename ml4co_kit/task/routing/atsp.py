r"""
Asymmetric Traveling Salesman Problem (ATSP)

ATSP is a variant of the classic TSP where the distance from 
one city to another may not be the same in both directions. 
It aims to find the shortest possible route that visits each city
once and returns to the starting point in a directed graph.
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
import networkx as nx
from typing import Union
from ml4co_kit.extension import tsplib95
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.base import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE


class ATSPTask(RoutingTaskBase):
    def __init__(
        self, 
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D, 
        round_type: ROUND_TYPE = ROUND_TYPE.NO, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(ATSPTask, self).__init__(
            task_type=TASK_TYPE.ATSP, 
            minimize=True,
            distance_type=distance_type,
            round_type=round_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = None  # Number of nodes
        self.dists = None  # Distance matrix
        
    def _normalize_dists(self):
        """Normalize dists to [0, 1] range."""
        dists = self.dists
        min_vals = np.min(dists)
        max_vals = np.max(dists)
        normalized_dists = (dists - min_vals) / (max_vals - min_vals)
        self.dists = normalized_dists
    
    def _check_dists_dim(self):
        """Check if dists are 2D or 3D."""
        if self.dists.ndim != 2 or self.dists.shape[0] != self.dists.shape[1]:
            raise ValueError(
                "dists should be a 2D array with shape (num_nodes, num_nodes)."
            )
    
    def _check_dists_not_none(self):
        """Check if dists are not None."""
        if self.dists is None:
            raise ValueError("``dists`` cannot be None!")
    
    def _check_sol_dim(self):
        """Ensure solution is a 1D array."""
        if self.sol.ndim != 1:
            raise ValueError("Solution should be a 1D array.")
    
    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 1D array."""
        if self.ref_sol.ndim != 1:
            raise ValueError("Reference solution should be a 1D array.")
    
    def from_data(
        self,
        dists: np.ndarray = None, 
        sol: np.ndarray = None, 
        ref: bool = False,
        normalize: bool = False,
        name: str = None
    ):
        # Set Attributes and Check Dimensions
        if dists is not None:
            self.dists = dists.astype(self.precision)
            self._check_dists_dim()
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()

        # Set Number of Nodes if Provided
        if self.dists is not None:
            self.nodes_num = self.dists.shape[0]

        # Normalize dists if Required
        if normalize:
            self._normalize_dists()
            
        # Set Name if Provided
        if name is not None:
            self.name = name
            
    def from_tsplib(
        self,
        atsp_file_path: pathlib.Path = None,
        tour_file_path: pathlib.Path = None,
        ref: bool = False,
        normalize: bool = False
    ):
        """Load ATSP data from a TSPLIB file."""
        # Read data from TSPLIB file if provided
        dists = name = None
        if atsp_file_path is not None:
            tsplib_data = tsplib95.load(atsp_file_path)
            name = tsplib_data.name
            try:
                dists = nx.to_numpy_array(tsplib_data.get_graph())
            except:
                try:
                    dists = np.array(tsplib_data.edge_weights)
                except:
                    raise RuntimeError(f"Error in loading {atsp_file_path}")
        
        # Read solution from tour file if provided
        sol = None
        if tour_file_path is not None:
            tsplib_data = tsplib95.load(tour_file_path)
            sol_list = list(tsplib_data.tours[0])
            sol_list.append(1)
            sol = np.array(sol_list) - 1
        
        # Use ``from_data``
        self.from_data(
            dists=dists, sol=sol, ref=ref, normalize=normalize, name=name
        )

    def to_tsplib(
        self, 
        atsp_file_path: pathlib.Path = None, 
        tour_file_path: pathlib.Path = None
    ):
        """Save ATSP data to a TSPLIB file."""
        # Save ATSP data to a TSPLIB file
        if atsp_file_path is not None:
            # Check data
            self._check_dists_not_none()
            dists = self.dists

            # Check file path
            check_file_path(atsp_file_path)
            
            # Write ATSP data to a TSPLIB file
            with open(atsp_file_path, "w") as f:
                f.write(f"NAME : {self.name}\n")
                f.write(f"COMMENT : Generated by ML4CO-Kit\n")
                f.write("TYPE : ATSP\n")
                f.write(f"DIMENSION : {self.nodes_num}\n")
                f.write(f"EDGE_WEIGHT_TYPE : EXPLICIT\n")
                f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
                f.write("EDGE_WEIGHT_SECTION:\n")
                for i in range(self.nodes_num):
                    line = ' '.join([str(elem) for elem in dists[i]])
                    f.write(f"{line}\n")
                f.write("EOF\n")
    
        # Save Solution
        if tour_file_path is not None:
            # Check data
            self._check_sol_not_none()
            sol = self.sol
            
            # Check file path
            check_file_path(tour_file_path)
            
            # Write Solution to a tour file
            with open(tour_file_path, "w") as f:
                f.write(f"NAME : {self.name}\n")
                f.write(f"TYPE: TOUR\n")
                f.write(f"DIMENSION : {self.nodes_num}\n")
                f.write(f"TOUR_SECTION\n")
                for i in range(self.nodes_num):
                    f.write(f"{sol[i] + 1}\n")
                f.write(f"-1\n")
                f.write(f"EOF\n")

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        if sol[0] != 0 or sol[-1] != 0:
            return False
        ordered_sol = np.sort(sol[1:])
        return True if np.all(ordered_sol == np.arange(self.nodes_num)) else False
    
    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the total distance of the TSP solution."""
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Evaluate
        total_distance = 0
        for i in range(len(sol) - 1):
            total_distance += self.dists[sol[i]][sol[i + 1]]
        return total_distance

    def render(self):
        """Render the ATSP problem instance with or without solution."""
        raise NotImplementedError("Render is not implemented for ATSP.")