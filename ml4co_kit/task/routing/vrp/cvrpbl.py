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
        mixed_backhaul: bool = False,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-4,
    ):
        # Super Initialization
        super(CVRPBLTask, self).__init__(
            cvrp_open=cvrp_open,
            distance_type=distance_type,
            round_type=round_type,
            precision=precision,
            threshold=threshold,
        )

        # Set Task Type and Mixed Backhaul
        self.task_type = TASK_TYPE.CVRPBL
        self.mixed_backhaul = mixed_backhaul

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

        # For each split tour, check:
        for split_idx in range(len(split_tours)):
            # Get the split tour
            split_tour: np.ndarray = split_tours[split_idx][1:]

            # Check the constraint B or MB
            if self.mixed_backhaul:
                if not self._check_route_mb(
                    route=split_tour,
                    demands=demands,
                    capacity=capacity,
                    threshold=self.threshold
                ):
                    return False
                if not self._check_route_b(
                    route=split_tour,
                    demands=demands,
                    capacity=capacity,
                    threshold=self.threshold
                ):
                    return False

            # Check the constraint L
            if not self._check_route_l(
                dist_eval=self.dist_eval,
                coords=self.coords,
                route=split_tour,
                max_route_length=self.max_route_length,
                threshold=self.threshold,
                cvrp_open=self.cvrp_open
            ):
                return False
        
        # If all constraints are satisfied, return True
        return True