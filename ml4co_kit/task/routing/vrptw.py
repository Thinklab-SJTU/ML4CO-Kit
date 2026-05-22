r"""
Vehicle Routing Problem with Time Windows (VRPTW).

VRPTW extends CVRP with customer time windows and service times. Early arrivals
can wait, while late arrivals are infeasible.
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
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class VRPTWTask(CVRPTask):
    def __init__(
        self,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(VRPTWTask, self).__init__(
            distance_type=distance_type,
            round_type=round_type,
            precision=precision,
            threshold=threshold
        )
        self.task_type = TASK_TYPE.VRPTW

        # Initialize Attributes
        self.time_windows = None           # Time windows for depots and points
        self.service_times = None          # Service times for depots and points

    def _check_time_windows_dim(self):
        """Ensure time_windows is a 2D array with shape (num_nodes + 1, 2)."""
        if self.time_windows.ndim != 2 or self.time_windows.shape[1] != 2:
            raise ValueError("Time windows should be a 2D array with shape (num_nodes + 1, 2).")
        if self.nodes_num is not None and self.time_windows.shape[0] != self.nodes_num + 1:
            raise ValueError("Time windows should include depot and all points.")

    def _check_time_windows_not_none(self):
        """Check if time_windows are not None."""
        if self.time_windows is None:
            raise ValueError("``time_windows`` cannot be None!")

    def _check_service_times_dim(self):
        """Ensure service_times is a 1D array with shape (num_nodes + 1,)."""
        if self.service_times.ndim != 1:
            raise ValueError("Service times should be a 1D array.")
        if self.nodes_num is not None and self.service_times.shape[0] != self.nodes_num + 1:
            raise ValueError("Service times should include depot and all points.")

    def _check_service_times_not_none(self):
        """Check if service_times are not None."""
        if self.service_times is None:
            raise ValueError("``service_times`` cannot be None!")

    def from_data(
        self,
        depots: np.ndarray = None,
        points: np.ndarray = None,
        demands: np.ndarray = None,
        capacity: float = None,
        time_windows: np.ndarray = None,
        service_times: np.ndarray = None,
        sol: np.ndarray = None,
        ref: bool = False,
        normalize: bool = False,
        name: str = None
    ):
        # Call Super Method ``from_data``
        super().from_data(
            depots=depots,
            points=points,
            demands=demands,
            capacity=capacity,
            sol=sol,
            ref=ref,
            normalize=normalize,
            name=name
        )
        if time_windows is not None:
            self.time_windows = time_windows.astype(self.precision)
            self._check_time_windows_dim()
        if service_times is not None:
            self.service_times = service_times.astype(self.precision)
            self._check_service_times_dim()

    def _check_route_time_windows(self, route: np.ndarray) -> bool:
        """Check time windows for a single route."""
        tw = self.time_windows
        service_times = self.service_times
        nodes = np.concatenate([np.array([0]), route.astype(int), np.array([0])])

        current_node = int(nodes[0])
        current_time = max(0.0, float(tw[current_node, 0]))
        if current_time > float(tw[current_node, 1]) + self.threshold:
            return False

        for idx in range(len(nodes) - 1):
            next_node = int(nodes[idx + 1])
            departure_time = current_time + float(service_times[current_node])
            travel_time = self.dist_eval.cal_distance(
                self.coords[current_node], self.coords[next_node]
            )
            arrival_time = departure_time + float(travel_time)

            if arrival_time > float(tw[next_node, 1]) + self.threshold:
                return False

            current_time = max(arrival_time, float(tw[next_node, 0]))
            current_node = next_node
        return True

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        if not super().check_constraints(sol):
            return False

        # Time Window Constraint
        self._check_time_windows_not_none()
        self._check_service_times_not_none()
        split_tours = np.split(sol, np.where(sol == 0)[0])[1: -1]
        for split_tour in split_tours:
            route = split_tour[1:]
            if not self._check_route_time_windows(route=route):
                return False
        return True
