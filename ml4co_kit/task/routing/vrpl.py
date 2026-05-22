r"""
Vehicle Routing Problem with Route Length Limit (VRPL).

VRPL extends CVRP by requiring every vehicle route to satisfy a maximum
route length constraint.
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


class VRPLTask(CVRPTask):
    def __init__(
        self,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(VRPLTask, self).__init__(
            distance_type=distance_type,
            round_type=round_type,
            precision=precision,
            threshold=threshold
        )
        self.task_type = TASK_TYPE.VRPL

        # Initialize Attributes
        self.max_route_length = None       # Maximum length for each route

    def _check_max_route_length_not_none(self):
        """Check if max_route_length is not None."""
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
        if max_route_length is not None:
            self.max_route_length = max_route_length

    def _evaluate_route_length(self, route: np.ndarray) -> np.floating:
        """Evaluate a single depot-delimited route."""
        route = np.concatenate([np.array([0]), route.astype(int), np.array([0])])
        route_length = 0
        for idx in range(len(route) - 1):
            cost = self.dist_eval.cal_distance(
                self.coords[route[idx]], self.coords[route[idx + 1]]
            )
            route_length += np.array(cost).astype(self.precision)
        return route_length

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid."""
        if not super().check_constraints(sol):
            return False

        # Route Length Constraint
        self._check_max_route_length_not_none()
        split_tours = np.split(sol, np.where(sol == 0)[0])[1: -1]
        for split_tour in split_tours:
            route = split_tour[1:]
            route_length = self._evaluate_route_length(route=route)
            if route_length > self.max_route_length + self.threshold:
                return False
        return True
