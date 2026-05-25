r"""
CVRP with backhauls and route length limit (VRPBL).
CVRPBL can be seen as a combination of CVRPB and CVRPL.
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


import numpy as np
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.vrp.cvrp import CVRPTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class CVRPBLTask(CVRPTask):
    def __init__(
        self,
        cvrp_open: bool = False,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5,
    ):
        # Super Initialization
        super(CVRPBLTask, self).__init__(
            cvrp_open=cvrp_open,
            distance_type=distance_type,
            round_type=round_type,
            precision=precision,
            threshold=threshold,
        )

        # Set Task Type
        self.task_type = TASK_TYPE.CVRPBL

        # Extra Attributes
        self.max_route_length = None # Maximum length for each route

    def _check_max_route_length_not_none(self):
        if self.max_route_length is None:
            raise ValueError("``max_route_length`` cannot be None!")

    def from_data(
        self,
        depots: np.ndarray = None,
        points: np.ndarray = None,
        demands: np.ndarray = None,
        capacity: float = None,
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

        # Set Maximum Route Length if Provided
        if max_route_length is not None:
            self.max_route_length = max_route_length

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

        # For each split tour, check:
        # 1. if the linehaul demand is served before the backhaul demand
        # 2. if the linehaul demand is within the capacity
        # 3. if the backhaul demand is within the capacity
        # 4. if the route length is within the maximum route length
        for split_idx in range(len(split_tours)):
            # Get the split tour and the demands on the split tour
            split_tour: np.ndarray = split_tours[split_idx][1:]
            route_demands = demands[split_tour.astype(int) - 1]

            # 1. Linehaul before backhaul
            is_linehaul = route_demands >= 0
            is_backhaul = route_demands < 0
            if np.any(is_backhaul):
                first_backhaul_idx = np.flatnonzero(is_backhaul)[0]
                if np.any(is_linehaul[first_backhaul_idx + 1:]):
                    return False

            # 2. Linehaul load <= vehicle capacity
            linehaul_load = np.sum(route_demands[is_linehaul], dtype=self.precision)
            if linehaul_load > capacity + self.threshold:
                return False

            # 3. Backhaul load <= vehicle capacity
            backhaul_load = np.sum(-route_demands[is_backhaul], dtype=self.precision)
            if backhaul_load > capacity + self.threshold:
                return False

            # 4. Route length <= maximum route length
            route_length = _eval_route_len(split_tour)
            if route_length > self.max_route_length + self.threshold:
                return False
        
        # If all constraints are satisfied, return True
        return True