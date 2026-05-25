r"""
CVRP with Length Limit and Time Windows (CVRPLTW).
CVRPLTW can be seen as a combination of CVRPL and CVRPTW.
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
from ml4co_kit.task.routing.vrp.cvrp import CVRPTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class CVRPLTWTask(CVRPTask):
    def __init__(
        self,
        cvrp_open: bool = False,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-4,
    ):
        # Super Initialization
        super(CVRPLTWTask, self).__init__(
            cvrp_open=cvrp_open,
            distance_type=distance_type,
            round_type=round_type,
            precision=precision,
            threshold=threshold,
        )

        # Set Task Type
        self.task_type = TASK_TYPE.CVRPLTW

        # Extra Attributes
        self.max_route_length = None  # Maximum length for each route
        self.tw = None         # Time windows (V+1, 2)
        self.service = None    # Service time (V+1,)

    def _check_max_route_length_not_none(self):
        if self.max_route_length is None:
            raise ValueError("``max_route_length`` cannot be None!")

    def _check_max_route_length_not_none(self):
        if self.max_route_length is None:
            raise ValueError("``max_route_length`` cannot be None!")

    def _check_tw_dim(self):
        if self.tw.ndim != 2 or self.tw.shape[1] != 2:
            raise ValueError(
                "Time windows should be a 2D array with shape (num_nodes, 2)."
            )

    def _check_tw_not_none(self):
        if self.tw is None:
            raise ValueError("``tw`` cannot be None!")

    def _check_service_dim(self):
        if self.service.ndim != 1:
            raise ValueError("Service service should be a 1D array.")

    def _check_service_not_none(self):
        if self.service is None:
            raise ValueError("``service`` cannot be None!")

    def from_data(
        self,
        depots: np.ndarray = None,
        points: np.ndarray = None,
        demands: np.ndarray = None,
        capacity: float = None,
        tw: np.ndarray = None,
        service: np.ndarray = None,
        max_route_length: float = None,
        sol: np.ndarray = None,
        ref: bool = False,
        normalize: bool = False,
        name: str = None,
    ):  
        # Call Super Method ``from_data``
        super().from_data(
            depots=depots, points=points, demands=demands, 
            capacity=capacity, sol=sol, ref=ref, 
            normalize=normalize, name=name
        )

        # Extra Attributes
        if max_route_length is not None:
            self.max_route_length = max_route_length
        if tw is not None:
            self.tw = tw.astype(self.precision)
            self._check_tw_dim()
        if service is not None:
            self.service = service.astype(self.precision)
            self._check_service_dim()

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        # Every tour starts and ends with the depot
        if sol[0] != 0 or sol[-1] != 0:
            return False
        
        # Every node is visited exactly once
        ordered_sol = np.sort(sol)[-self.nodes_num:]
        if not np.all(ordered_sol == (np.arange(self.nodes_num) + 1)):
            return False

        # Split Tours
        demands = self.demands
        capacity = self.capacity
        split_tours = self._split_tours(sol)

        # Define a helper function to evaluate the route length
        def _eval_route_len(route: np.ndarray) -> np.floating:
            # Initialize the route length
            route_length = 0

            # Depot -> First Customer
            route_length += self.dist_eval.cal_distance(
                self.coords[0], self.coords[route[0]]
            )

            # Customers in the route
            for idx in range(len(route) - 1):
                route_length += self.dist_eval.cal_distance(
                    self.coords[route[idx]], self.coords[route[idx + 1]]
                )

            # If not cvrp_open, add the distance from the last customer to the depot
            if not self.cvrp_open:
                route_length += self.dist_eval.cal_distance(
                    self.coords[route[-1]], self.coords[0]
                )

            # Return the route length
            return route_length

        # Define a helper function to check the route time windows
        def _check_route_tw(route: np.ndarray) -> bool:
            # Initialize the current time
            cur_time: float = 0
            last_node = 0

            # Sequentially check each node in the route
            for node in route.astype(int):
                # Get the time window and service time
                cur_tw = self.tw[node]
                cur_tw_0 = float(cur_tw[0])
                cur_tw_1 = float(cur_tw[1])
                cur_st = self.service[node]

                # Update the current time
                travel_time = self.dist_eval.cal_distance(
                    self.coords[last_node], self.coords[node]
                )
                cur_time += float(travel_time)
                cur_time = cur_tw_0 if cur_time < cur_tw_0 else cur_time

                # Check if the current time is within the time window
                if cur_time > cur_tw_1 + self.threshold:
                    return False 
                
                # Update the current time with the service time
                cur_time += cur_st

                # Update the last node
                last_node = node

            # If not open, final check (back to depot)
            if not self.cvrp_open:
                # Get the time window and service time
                cur_tw: np.ndarray = self.tw[0]
                cur_tw_1: float = float(cur_tw[1])

                # Update the current time
                travel_time = self.dist_eval.cal_distance(
                    self.coords[last_node], self.coords[0]
                )
                cur_time += float(travel_time)
                
                # Check if the current time is within the time window
                if cur_time > cur_tw_1 + self.threshold:
                    return False

            # Return True if the route time windows are satisfied
            return True

        # For each split tour, check: 
        # 1. if the demand is within the capacity
        # 2. if the route length is within the maximum route length
        # 3. if the route time windows are satisfied
        for split_idx in range(len(split_tours)):
            # Get the split tour and the demands on the split tour
            split_tour: np.ndarray = split_tours[split_idx][1:]
            route_demands = demands[split_tour.astype(int) - 1]

            # 1. if the demand is within the capacity
            split_demand_need = np.sum(route_demands, dtype=self.precision)
            if split_demand_need > capacity + self.threshold:
                return False

            # 2. if the route length is within the maximum route length
            route_length = _eval_route_len(split_tour)
            if route_length > self.max_route_length + self.threshold:
                return False

            # 3. if the route time windows are satisfied
            if not _check_route_tw(split_tour):
                return False

        # If all constraints are satisfied, return True
        return True

    def render(
        self, 
        save_path: pathlib.Path, 
        with_sol: bool = True,
        figsize: tuple = (5, 5),
        node_color: str = "darkblue",
        edge_color: str = "darkblue",
        node_size: int = 50,
    ):
        pass
        # TODO: Implement the render method for CVRPLTW