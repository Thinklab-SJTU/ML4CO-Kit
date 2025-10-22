r"""
Capacitated Vehicle Routing Problem (CVRP). 

The CVRP problems requires finding the most efficient routes for a fleet of vehicles
with limited capacity to deliver goods to a set of customers while minimizing costs.
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


import vrplib
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.base import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE


class CVRPTask(RoutingTaskBase):
    def __init__(
        self, 
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D, 
        round_type: ROUND_TYPE = ROUND_TYPE.NO, 
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super().__init__(
            task_type=TASK_TYPE.CVRP, 
            minimize=True,
            distance_type=distance_type,
            round_type=round_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = None              # Number of nodes (besides depots)
        self.depots = None                 # Coordinates of depots
        self.points = None                 # Coordinates of points
        self.coords = None                 # All coordinates (including depots and points)
        self.demands = None                # Demands of points
        self.norm_demands = None           # Normalized demands of points
        self.capacity = None               # Capacity of vehicles
        self.dists = None                  # Distance matrix
        self.threshold = threshold         # Threshold for floating point precision
  
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
            
    def _check_demands_dim(self):
        """Ensure demands is a 1D array."""
        if self.demands.ndim != 1:
            raise ValueError("Demands should be a 1D array.")
    
    def _check_demands_not_none(self):
        """Check if demands are not None."""
        if self.demands is None:
            raise ValueError("``demands`` cannot be None!")
    
    def _check_norm_demands_not_none(self):
        """Check if norm_demands are not None."""
        if self.norm_demands is None:
            raise ValueError(
                "``norm_demands`` cannot be None! This attribute is generated  "
                "automatically when ``demands`` and ``capacity`` are provided."
            )
    
    def _check_capacity_not_none(self):
        """Check if capacity is not None."""
        if self.capacity is None:
            raise ValueError("``capacity`` cannot be None!")
    
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
        demands: np.ndarray = None,
        capacity: float = None,
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
        if demands is not None:
            self.demands = demands.astype(self.precision)
            self._check_demands_dim()
        if capacity is not None:
            self.capacity = capacity
        
        # Merge depots and points
        if self.depots is not None and self.points is not None:
            self.coords = np.concatenate(
                [np.expand_dims(self.depots, axis=0), self.points], axis=0
            )
        
        # Normalize demands accroding to capacity
        if self.demands is not None and self.capacity is not None:
            self.norm_demands = self.demands / self.capacity

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
            
    def from_vrplib(
        self, 
        vrp_file_path: pathlib.Path = None, 
        sol_file_path: pathlib.Path = None,
        ref: bool = False,
        normalize: bool = False
    ):
        """Load CVRP data from a VRPLIB file."""
        # Read data from TSPLIB file if provided
        points = depots = demands = capacity = name =None
        if vrp_file_path is not None: 
            # Load CVRP data from VRPLIB file
            instance = vrplib.read_instance(
                path=vrp_file_path, compute_edge_weights=False
            )
            name = instance["name"]
            coords = instance["node_coord"]
            depots = coords[0]
            points = coords[1:]
            capacity = instance["capacity"]
            demands = instance["demand"][1:]
            
        # Read solution from sol file if provided
        sol = None
        if sol_file_path is not None:
            route_flag = None
            with open(sol_file_path, "r") as file:
                first_line = file.readline()
                if "Route" in first_line:
                    # Like this
                    # Route #1: 15 17 9 3 16 29
                    # Route #2: 12 5 26 7 8 13 32 2
                    route_flag = True
                else:
                    # Like this
                    # 36395
                    # 37
                    # 1893900
                    # 1133620
                    # 0 1 1 1144 12  14 0 217 236 105 2 169 8 311 434 362 187 136 59 0
                    # 0 1 1 1182 12  14 0 3 370 133 425 349 223 299 386 267 410 411 348 0
                    route_flag = False
            
            # Deal with different cases
            if route_flag == True:
                with open(sol_file_path, "r") as file:
                    tour = [0]
                    for line in file:
                        if line.startswith("Route"):
                            split_line = line.split(" ")[2:]
                            for node in split_line:
                                if node != "\n":
                                    tour.append(int(node))
                            tour.append(0)
            elif route_flag == False:
                with open(sol_file_path, "r") as file:
                    line_idx = 0
                    tour = list()
                    for line in file:
                        line_idx += 1
                        if line_idx < 5:
                            continue
                        split_line = line.split(" ")[7:-1][1:]
                        for node in split_line:
                            tour.append(int(node))
                    tour.append(0)
            else:
                raise ValueError(
                    f"Unable to read route information from {sol_file_path}."
                )
            sol = np.array(tour)
            
        # Use ``from_data``
        self.from_data(
            depots=depots, points=points, demands=demands, capacity=capacity, 
            sol=sol, ref=ref, normalize=normalize, name=name
        )
    
    def to_vrplib(
        self, 
        vrp_file_path: pathlib.Path = None, 
        sol_file_path: pathlib.Path = None
    ):
        """Save CVRP data to a VRPLIB file."""
        # Save CVRP data to a VRPLIB file
        if vrp_file_path is not None:
            # Check data
            self._check_depots_not_none()
            self._check_points_not_none()
            self._check_demands_not_none()
            self._check_capacity_not_none()
            depots = self.depots
            points = self.points
            demands = self.demands
            capacity = self.capacity
            
            # Check file path
            check_file_path(vrp_file_path)

            # Write CVRP data to a VRPLIB file
            with open(vrp_file_path, "w") as f:
                f.write(f"NAME : {vrp_file_path}\n")
                f.write(f"COMMENT : Generated by ML4CO-Kit\n")
                f.write("TYPE : CVRP\n")
                f.write(f"DIMENSION : {self.nodes_num + 1}\n")
                f.write(f"EDGE_WEIGHT_TYPE : {self.distance_type.value}\n")
                f.write(f"CAPACITY : {capacity}\n")
                f.write("NODE_COORD_SECTION\n")
                x, y = depots
                f.write(f"1 {x} {y}\n")
                for i in range(self.nodes_num):
                    x, y = points[i]
                    f.write(f"{i+2} {x} {y}\n")
                f.write("DEMAND_SECTION \n")
                f.write(f"1 0\n")
                for i in range(self.nodes_num):
                    f.write(f"{i+2} {demands[i]}\n")
                f.write("DEPOT_SECTION \n")
                f.write("	1\n")
                f.write("	-1\n")
                f.write("EOF\n")
        
        # Save Solution to a sol file
        if sol_file_path is not None:
            # Check data
            self._check_sol_not_none()
            sol = self.sol            

            # Check file path
            check_file_path(sol_file_path)            

            # Use ``evaluate`` to get cost
            cost = self.evaluate(sol)
            
            # Write Solution to a sol file
            split_tours = np.split(sol, np.where(sol == 0)[0])[1: -1]
            with open(sol_file_path, "w") as f:
                for i in range(len(split_tours)):
                    part_tour = split_tours[i][1:]
                    f.write(f"Route #{i+1}: ")
                    f.write(f" ".join(str(int(node)) for node in part_tour))
                    f.write("\n")
                f.write(f"Cost {cost}\n")

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        # Every tour starts and ends with the depot
        if sol[0] != 0 or sol[-1] != 0:
            return False
        
        # Every node is visited exactly once
        ordered_sol = np.sort(sol)[-self.nodes_num:]
        if not np.all(ordered_sol == (np.arange(self.nodes_num) + 1)):
            return False
        
        # Demands Constraint
        demands = self.demands
        capacity = self.capacity
        split_tours = np.split(sol, np.where(sol == 0)[0])[1: -1]
        for split_idx in range(len(split_tours)):   
            split_tour = split_tours[split_idx][1:]
            split_demand_need = np.sum(demands[split_tour.astype(int) - 1], dtype=np.float32)
            if split_demand_need > capacity + self.threshold:
                return False
        return True
        
    def evaluate(self, sol: np.ndarray) -> np.floating:
        """Evaluate the total distance of the CVRP solution."""
        # Check Constraints
        if not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Evaluate
        total_distance = 0
        for i in range(len(sol) - 1):
            cost = self.dist_eval.cal_distance(
                self.coords[sol[i]], self.coords[sol[i + 1]]
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
        """Render the CVRP problem instance with or without solution."""
        
        # Check ``save_path``
        check_file_path(save_path)
        
        # Get Attributes
        depots = self.depots
        points = self.points
        sol = self.sol

        # Draw Graph
        if with_sol:
            _, ax = plt.subplots(figsize=figsize)
            kwargs = dict(c="tab:red", marker="*", zorder=3, s=500)
            ax.scatter(depots[0], depots[1], label="Depot", **kwargs)

            coords = np.concatenate([np.expand_dims(depots, axis=0), points], axis=0)
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            split_tours = np.split(sol, np.where(sol == 0)[0])[1: -1]
            idx = 0
            for part_tour in split_tours:
                route = part_tour[1:]
                x = x_coords[route]
                y = y_coords[route]

                # Coordinates of clients served by this route.
                if len(route) == 1:
                    ax.scatter(x, y, label=f"Route {idx}", zorder=3, s=node_size)
                ax.plot(x, y)
                arrowprops = dict(arrowstyle='->', linewidth=0.25, color='grey')
                ax.annotate(
                    text='', 
                    xy=(x_coords[0], y_coords[0]), 
                    xytext=(x[0], y[0]), 
                    arrowprops=arrowprops
                )
                ax.annotate(
                    text='', 
                    xy=(x[-1], y[-1]), 
                    xytext=(x_coords[0], y_coords[0]), 
                    arrowprops=arrowprops
                )
            
            ax.set_aspect("equal", "datalim")
            ax.legend(frameon=False, ncol=2)            
        else:
            _, ax = plt.subplots(figsize=figsize)
            kwargs = dict(c="tab:red", marker="*", zorder=3, s=500)
            ax.scatter(depots[0], depots[1], label="Depot", **kwargs)
            ax.scatter(points[:, 0], points[:, 1], s=node_size, label="Clients")
            ax.grid(color="grey", linestyle="solid", linewidth=0.2)
            ax.set_aspect("equal", "datalim")
            ax.legend(frameon=False, ncol=2)

        # Save Figure
        plt.savefig(save_path)