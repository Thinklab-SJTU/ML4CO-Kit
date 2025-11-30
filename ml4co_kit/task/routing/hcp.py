r"""
Hamiltonian Cycle Problem (HCP).

HCP requires finding a cycle that visits each vertex of the graph exactly once 
and returns to the starting node. Unlike TSP, HCP works on arbitrary graph structures
and is a decision problem rather than an optimization problem.
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


class HCPTask(RoutingTaskBase):
    def __init__(
        self, 
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D, 
        round_type: ROUND_TYPE = ROUND_TYPE.NO, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.HCP, 
            minimize=True,
            distance_type=distance_type,
            round_type=round_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = None              # Number of nodes
        self.points = None                 # Coordinates of points (optional)
        self.adj_matrix = None             # Adjacency matrix of the graph
        self.graph = None                  # NetworkX graph object
    
    def _check_adj_matrix_dim(self):
        """Check if adjacency matrix is square."""
        if self.adj_matrix.ndim != 2 or self.adj_matrix.shape[0] != self.adj_matrix.shape[1]:
            raise ValueError(
                "Adjacency matrix should be a square 2D array."
            )
    
    def _check_adj_matrix_not_none(self):
        r"""
        Checks if the ``adj_matrix`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``adj_matrix`` is ``None``. 
        """
        if self.adj_matrix is None:
            raise ValueError("``adj_matrix`` cannot be None for HCP!")
    
    def _check_points_dim(self):
        """Check if points are 2D or 3D."""
        if self.points is not None and (self.points.ndim != 2 or self.points.shape[1] not in [2, 3]):
            raise ValueError(
                "Points should be a 2D array with shape (num_points, 2) or (num_points, 3)."
            )
    
    def _check_sol_dim(self):
        """Ensure solution is a 1D array."""
        if self.sol.ndim != 1:
            raise ValueError("Solution should be a 1D array.")

    def _check_ref_sol_dim(self):
        """Ensure reference solution is a 1D array."""
        if self.ref_sol.ndim != 1:
            raise ValueError("Reference solution should be a 1D array.")

    def _build_graph_from_adj_matrix(self):
        """Build NetworkX graph from adjacency matrix."""
        if self.adj_matrix is not None and self.graph is None:
            self.graph = nx.from_numpy_array(self.adj_matrix)
    
    def from_data(
        self,
        adj_matrix: np.ndarray = None,
        points: np.ndarray = None, 
        sol: np.ndarray = None, 
        ref: bool = False,
        name: str = None
    ):
        # Set Attributes and Check Dimensions
        if adj_matrix is not None:
            self.adj_matrix = adj_matrix.astype(int)
            self._check_adj_matrix_dim()
        if points is not None:
            self.points = points.astype(self.precision)
            self._check_points_dim()
        
        # Check that adjacency matrix is provided
        self._check_adj_matrix_not_none()
        
        # Set Number of Nodes
        self.nodes_num = self.adj_matrix.shape[0]
        
        # Build graph representation
        self._build_graph_from_adj_matrix()
        
        # Set Solution if Provided
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
  
    def from_tsplib(
        self, 
        hcp_file_path: pathlib.Path = None, 
        tour_file_path: pathlib.Path = None,
        ref: bool = False
    ):
        """Load HCP data from a TSPLIB file."""
        # Read data from TSPLIB file if provided
        adj_matrix = name = None
        if hcp_file_path is not None:
            tsplib_data = tsplib95.load(hcp_file_path)
            name = tsplib_data.name
            try:
                graph = tsplib_data.get_graph()
                adj_matrix = nx.to_numpy_array(graph)
            except:
                try:
                    # Try to extract from edge weights
                    adj_matrix = np.array(tsplib_data.edge_weights)
                    if adj_matrix.ndim == 1:
                        # Convert to matrix if needed
                        n = int(np.sqrt(len(adj_matrix)))
                        adj_matrix = adj_matrix.reshape(n, n)
                except:
                    raise RuntimeError(f"Error in loading {hcp_file_path}")
               
        # Read solution from tour file if provided
        sol = None  
        if tour_file_path is not None:
            tsp_tour = tsplib95.load(tour_file_path)
            tsp_tour = tsp_tour.tours
            tsp_tour: list
            tsp_tour = tsp_tour[0]
            tsp_tour.append(tsp_tour[0])  # Ensure cycle closes
            sol = np.array(tsp_tour) - 1

        # Use ``from_data``
        self.from_data(
            adj_matrix=adj_matrix, sol=sol, ref=ref, name=name
        )

    def to_tsplib(
        self, 
        hcp_file_path: pathlib.Path = None, 
        tour_file_path: pathlib.Path = None
    ):
        """Save HCP data to a TSPLIB file."""
        # Save HCP data to a TSPLIB file
        if hcp_file_path is not None:
            # Check data
            self._check_adj_matrix_not_none()
            adj_matrix = self.adj_matrix

            # Check file path
            check_file_path(hcp_file_path)
            
            # Write HCP data to a TSPLIB file
            with open(hcp_file_path, "w") as f:
                f.write(f"NAME : {self.name}\n")
                f.write(f"COMMENT : Generated by ML4CO-Kit\n")
                f.write("TYPE : HCP\n")
                f.write(f"DIMENSION : {self.nodes_num}\n")
                f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
                f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
                f.write("EDGE_WEIGHT_SECTION\n")
                for i in range(self.nodes_num):
                    line = ' '.join([str(int(elem)) for elem in adj_matrix[i]])
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
                f.write(f"DIMENSION: {self.nodes_num}\n")
                f.write(f"TOUR_SECTION\n")
                for i in range(len(sol)):
                    f.write(f"{sol[i] + 1}\n")
                f.write(f"-1\n")
                f.write(f"EOF\n")

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is a valid Hamiltonian cycle."""
        # Check if solution forms a cycle
        if sol[0] != sol[-1]:
            return False
        
        # Check if all nodes are visited exactly once (except the repeated start/end)
        visited_nodes = sol[:-1]
        if len(np.unique(visited_nodes)) != self.nodes_num:
            return False
        
        # Check if all edges exist in the graph
        if self.adj_matrix is not None:
            for i in range(len(sol) - 1):
                if self.adj_matrix[sol[i], sol[i + 1]] == 0:
                    return False
        
        return True
    
    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the Hamiltonian cycle (feasibility problem)."""
        # For HCP, we return 0 for feasible solutions, 1 for infeasible
        if self.check_constraints(sol):
            return np.array(0.0, dtype=self.precision)
        else:
            return np.array(1.0, dtype=self.precision)

    def render(
        self, 
        save_path: pathlib.Path, 
        with_sol: bool = True,
        figsize: tuple = (5, 5),
        node_color: str = "darkblue",
        edge_color: str = "darkblue",
        node_size: int = 50,
    ):
        """Render the HCP problem instance with or without solution."""
        
        # Check ``save_path``
        check_file_path(save_path)
        
        # Get Attributes
        points = self.points
        sol = self.sol
        
        # Create graph from adjacency matrix
        if self.graph is not None:
            graph = self.graph
        else:
            graph = nx.from_numpy_array(self.adj_matrix)
        
        # Position nodes
        if points is not None:
            pos = dict(zip(range(len(points)), points.tolist()))
        else:
            pos = nx.spring_layout(graph, seed=42)
        
        # Edge highlighting for solution
        edge_list = []
        if with_sol:
            if sol is None:
                raise ValueError("Solution is not provided!")
            for i in range(len(sol) - 1):
                edge_list.append((sol[i], sol[i + 1]))

        # Draw Graph
        figure = plt.figure(figsize=figsize)
        figure.add_subplot(111)
        
        # Draw all possible edges in light gray
        nx.draw_networkx_edges(
            G=graph, pos=pos, alpha=0.1, width=0.5, edge_color="lightgray"
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G=graph, pos=pos, node_color=node_color, node_size=node_size
        )
        
        # Draw solution edges if available
        if with_sol and edge_list:
            nx.draw_networkx_edges(
                G=graph, pos=pos, edgelist=edge_list, alpha=1, width=2, edge_color=edge_color
            )
        
        # Draw node labels
        nx.draw_networkx_labels(G=graph, pos=pos, font_size=8)

        # Save Figure
        plt.savefig(save_path)
        plt.close()

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