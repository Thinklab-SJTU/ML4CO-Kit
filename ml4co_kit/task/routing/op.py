r"""
Orienteering Problem (OP).

The Orienteering Problem requires finding a path that maximizes the total prize
collected while respecting a given time or distance max_length constraint.
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
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.base import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE


class OPTask(RoutingTaskBase):
    def __init__(
        self, 
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D, 
        round_type: ROUND_TYPE = ROUND_TYPE.NO, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.OP, 
            minimize=False,  # OP is a maximization problem
            distance_type=distance_type,
            round_type=round_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = None              # Number of nodes
        self.depots = None                 # Coordinates of depots
        self.points = None                 # Coordinates of points
        self.coords = None                 # All coordinates (including depots and points)
        self.prizes = None                 # Prize values for each node
        self.max_length = None             # Maximum length of the path
        self.dists = None                  # Distance matrix
    
    def _normalize_depots_and_points(self):
        """Normalize depots and points to [0, 1] range."""
        depots = self.depots
        points = self.points
        min_vals = min(np.min(points), np.min(self.depots))
        max_vals = max(np.max(points), np.max(self.depots))
        normalized_points = (points - min_vals) / (max_vals - min_vals)
        normalized_depots = (depots - min_vals) / (max_vals - min_vals)
        self.points = normalized_points
        self.depots = normalized_depots
    
    def _check_depots_dim(self):
        """Check if depots are 1D or 2D."""
        if self.depots.ndim != 1 or self.depots.shape[0] not in [2, 3]:
            raise ValueError(
                "Depots should be a 1D array with shape (2,) or (3,)."
            )
            
    def _check_depots_not_none(self):
        """Check if depots are not None."""
        if self.depots is None:
            raise ValueError("``depots`` cannot be None!")
    
    def _check_points_dim(self):
        """Check if points are 2D or 3D."""
        if self.points.ndim != 2 or self.points.shape[1] not in [2, 3]:
            raise ValueError(
                "Points should be a 2D array with shape (num_points, 2) or (num_points, 3)."
            )
    
    def _check_points_not_none(self):
        """Check if points are not None."""
        if self.points is None:
            raise ValueError("``points`` cannot be None!")
        
    def _check_coords_not_none(self):
        """Check if coords are not None."""
        if self.coords is None:
            raise ValueError(
                "``coords`` cannot be None! This attribute is generated "
                "automatically when ``depots`` and ``points`` are provided."
            )
    
    def _check_prizes_dim(self):
        """Ensure prizes is a 1D array."""
        if self.prizes.ndim != 1:
            raise ValueError("Prizes should be a 1D array.")
    
    def _check_prizes_not_none(self):
        """Check if prizes are not None."""
        if self.prizes is None:
            raise ValueError("``prizes`` cannot be None!")
    
    def _check_max_length_not_none(self):
        """Check if max_length is not None."""
        if self.max_length is None:
            raise ValueError("``max_length`` cannot be None!")
    
    def _check_sol_dim(self):
        """Ensure solution is a 1D array."""
        if self.sol.ndim != 1:
            raise ValueError("Solution should be a 1D array.")

    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 1D array."""
        if self.ref_sol.ndim != 1:
            raise ValueError("Reference solution should be a 1D array.")

    def _get_dists(self) -> np.ndarray:
        """Get distance matrix."""
        if self.dists is None:
            dists = np.zeros((self.nodes_num + 1, self.nodes_num + 1))
            for i in range(self.nodes_num + 1):
                for j in range(i + 1, self.nodes_num + 1):
                    dists[i, j] = self.dist_eval.cal_distance(self.coords[i], self.coords[j])
                    dists[j, i] = dists[i, j]
            self.dists = dists.astype(self.precision)
        return self.dists

    def from_data(
        self,
        depots: np.ndarray = None,
        points: np.ndarray = None, 
        prizes: np.ndarray = None,
        max_length: float = None,
        sol: np.ndarray = None, 
        ref: bool = False,
        normalize: bool = False,
        name: str = None
    ):
        # Set Attributes and Check Dimensions
        if depots is not None:
            self.depots = depots.astype(self.precision)
            self._check_depots_dim() 
        if points is not None:
            self.points = points.astype(self.precision)
            self._check_points_dim()
        if prizes is not None:
            self.prizes = prizes.astype(self.precision)
            self._check_prizes_dim()
        if max_length is not None:
            self.max_length = max_length

        # Merge depots and points
        if self.depots is not None and self.points is not None:
            self.coords = np.concatenate(
                [np.expand_dims(self.depots, axis=0), self.points], axis=0
            )

        # Set Solution if Provided
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()
        
        # Normalize Depots and Points if Required
        if normalize:
            self._normalize_depots_and_points()
        
        # Set Number of Nodes if Provided
        if self.points is not None:
            self.nodes_num = self.points.shape[0]
            
        # Set Name if Provided
        if name is not None:
            self.name = name
  
    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        # Every tour starts and ends with the depot
        if sol[0] != 0 or sol[-1] != 0:
            return False
        
        # Check max_length constraint
        total_distance = 0
        for i in range(len(sol) - 1):
            total_distance += self.dist_eval.cal_distance(
                self.coords[sol[i]], self.coords[sol[i + 1]]
            )
        if total_distance > self.max_length:
            return False
            
        return True
    
    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the total prize collected by the OP solution."""
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Calculate total prize collected
        total_prize = np.sum(self.prizes[sol[1:-1]-1])       
        return total_prize
    
    def render(
        self, 
        save_path: pathlib.Path, 
        with_sol: bool = True,
        figsize: tuple = (5, 5),
        node_color: str = "darkblue",
        edge_color: str = "darkblue",
        node_size: int = 50,
    ):
        """Render the OP problem instance with or without solution."""
        raise NotImplementedError("Render is not implemented for OP.")