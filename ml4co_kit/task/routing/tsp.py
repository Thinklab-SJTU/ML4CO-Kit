r"""
Traveling Salesman Problem (TSP).

TSP requires finding the shortest tour that visits each vertex 
of the graph exactly once and returns to the starting node. 
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
import matplotlib.pyplot as plt
from typing import Union
from ml4co_kit.extension import tsplib95
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.base import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE


class TSPTask(RoutingTaskBase):
    def __init__(
        self, 
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D, 
        round_type: ROUND_TYPE = ROUND_TYPE.NO, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.TSP, 
            minimize=True,
            distance_type=distance_type,
            round_type=round_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = None              # Number of nodes
        self.points = None                 # Coordinates of points
        self.dists = None                  # Distance matrix
    
    def _normalize_points(self):
        """Normalize points to [0, 1] range."""
        points = self.points
        min_vals = np.min(points)
        max_vals = np.max(points)
        normalized_points = (points - min_vals) / (max_vals - min_vals)
        self.points = normalized_points
    
    def _check_points_dim(self):
        """Check if points are 2D or 3D."""
        if self.points.ndim != 2 or self.points.shape[1] not in [2, 3]:
            raise ValueError(
                "Points should be a 2D array with shape (num_points, 2) or (num_points, 3)."
            )
    
    def _check_points_not_none(self):
        r"""
        Checks if the ``points`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``points`` is ``None``. 
        """
        if self.points is None:
            raise ValueError("``points`` cannot be None!")
    
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
            dists = np.zeros((self.nodes_num, self.nodes_num))
            for i in range(self.nodes_num):
                for j in range(i + 1, self.nodes_num):
                    dists[i, j] = self.dist_eval.cal_distance(self.points[i], self.points[j])
                    dists[j, i] = dists[i, j]
            self.dists = dists.astype(self.precision)
        return self.dists
    
    def from_data(
        self,
        points: np.ndarray = None, 
        sol: np.ndarray = None, 
        ref: bool = False,
        normalize: bool = False,
        name: str = None
    ):
        # Set Attributes and Check Dimensions
        if points is not None:
            self.dists = None
            self.points = points.astype(self.precision)
            self._check_points_dim()
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()
        
        # Set Number of Nodes if Provided
        if self.points is not None:
            self.nodes_num = self.points.shape[0]
        
        # Normalize Points if Required
        if normalize:
            self._normalize_points()
            
        # Set Name if Provided
        if name is not None:
            self.name = name
  
    def from_tsplib(
        self, 
        tsp_file_path: pathlib.Path = None, 
        tour_file_path: pathlib.Path = None,
        ref: bool = False,
        normalize: bool = False
    ):
        """Load TSP data from a TSPLIB file."""
        # Read data from TSPLIB file if provided
        points = name = None
        if tsp_file_path is not None:
            tsplib_data = tsplib95.load(tsp_file_path)
            name = tsplib_data.name
            self.distance_type = DISTANCE_TYPE(tsplib_data.edge_weight_type)
            points = np.array(list(tsplib_data.node_coords.values()))
               
        # Read solution from tour file if provided
        sol = None  
        if tour_file_path is not None:
            tsp_tour = tsplib95.load(tour_file_path)
            tsp_tour = tsp_tour.tours
            tsp_tour: list
            tsp_tour = tsp_tour[0]
            tsp_tour.append(1)
            sol = np.array(tsp_tour) - 1

        # Use ``from_data``
        self.from_data(
            points=points, sol=sol, ref=ref, normalize=normalize, name=name
        )

    def to_tsplib(
        self, 
        tsp_file_path: pathlib.Path = None, 
        tour_file_path: pathlib.Path = None
    ):
        """Save TSP data to a TSPLIB file."""
        # Save TSP data to a TSPLIB file
        if tsp_file_path is not None:
            # Check data
            self._check_points_not_none()
            points = self.points

            # Check file path
            check_file_path(tsp_file_path)
            
            # Write TSP data to a TSPLIB file
            with open(tsp_file_path, "w") as f:
                f.write(f"NAME : {self.name}\n")
                f.write(f"COMMENT : Generated by ML4CO-Kit\n")
                f.write("TYPE : TSP\n")
                f.write(f"DIMENSION : {self.nodes_num}\n")
                f.write(f"EDGE_WEIGHT_TYPE : {self.distance_type.value}\n")
                f.write("NODE_COORD_SECTION\n")
                for i in range(self.nodes_num):
                    x, y = points[i]
                    f.write(f"{i+1} {x} {y}\n")
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
                f.write(f"DIMENSION: {self.nodes_num}\n")
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
            cost = self.dist_eval.cal_distance(
                self.points[sol[i]], self.points[sol[i + 1]]
            )
            total_distance += np.array(cost).astype(self.precision)
        return total_distance

    def render(
        self, 
        save_path: pathlib.Path, 
        with_sol: bool = True,
        figsize: tuple = (5, 5),
        node_color: str = "darkblue",
        edge_color: str = "darkblue",
        node_size: int = 50,
    ):
        """Render the TSP problem instance with or without solution."""
        
        # Check ``save_path``
        check_file_path(save_path)
        
        # Get Attributes
        points = self.points
        sol = self.sol
        
        # Edge Values
        edge_values = (
            np.sum(
                (np.expand_dims(points, 1) - np.expand_dims(points, 0)) ** 2, axis=-1
            )
            ** 0.5
        )

        # Edge Target
        nodes_num = points.shape[0]
        edge_target = np.zeros((nodes_num, nodes_num))
        if with_sol:
            if sol is None:
                raise ValueError("Solution is not provided!")
            for i in range(len(sol) - 1):
                edge_target[sol[i], sol[i + 1]] = 1
        target_pairs = self.edges_to_node_pairs(edge_target)
        graph = nx.from_numpy_array(edge_values)
        pos = dict(zip(range(len(points)), points.tolist()))

        # Draw Graph
        figure = plt.figure(figsize=figsize)
        figure.add_subplot(111)
        nx.draw_networkx_nodes(G=graph, pos=pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(
            G=graph, pos=pos, edgelist=target_pairs, alpha=1, width=1, edge_color=edge_color
        )

        # Save Figure
        plt.savefig(save_path)

    @staticmethod
    def edges_to_node_pairs(edge_target: np.ndarray):
        r"""
        Helper function to convert edge matrix into pairs of adjacent nodes.
        """
        pairs = []
        for r in range(len(edge_target)):
            for c in range(len(edge_target)):
                if edge_target[r][c] == 1:
                    pairs.append((r, c))
        return pairs