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

        # For each split tour, check: 
        for split_idx in range(len(split_tours)):
            # Get the split tour
            split_tour: np.ndarray = split_tours[split_idx][1:]
            
            # Check the constraint C
            if not self._check_route_c(
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
            
            # Check the constraint TW
            if not self._check_route_tw(
                dist_eval=self.dist_eval,
                coords=self.coords,
                route=split_tour,
                tw=self.tw,
                service=self.service,
                threshold=self.threshold,
                cvrp_open=self.cvrp_open
            ):
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