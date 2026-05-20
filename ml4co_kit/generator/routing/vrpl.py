r"""
Generator for VRPL instances.
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
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.vrpl import VRPLTask
from ml4co_kit.generator.routing.cvrp import CVRPGenerator, CVRP_TYPE
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class VRPL_TYPE(str, Enum):
    """Define the VRPL types as an enumeration."""
    UNIFORM = "uniform" # Uniform coords
    GAUSSIAN = "gaussian" # Gaussian coords


class VRPLGenerator(CVRPGenerator):
    """Generator for Vehicle Routing Problem with Route Length Limit instances."""

    def __init__(
        self,
        distribution_type: VRPL_TYPE = VRPL_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for demands and capacity
        min_demand: int = 1,
        max_demand: int = 9,
        min_capacity: int = 40,
        max_capacity: int = 40,
        # special args for route length
        max_route_length: float = None,
        # special args for gaussian
        gaussian_mean_x: float = 0.0,
        gaussian_mean_y: float = 0.0,
        gaussian_std: float = 1.0,
    ):
        # Super Initialization
        super(VRPLGenerator, self).__init__(
            distribution_type=CVRP_TYPE(distribution_type.value),
            precision=precision,
            nodes_num=nodes_num,
            min_demand=min_demand,
            max_demand=max_demand,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            gaussian_mean_x=gaussian_mean_x,
            gaussian_mean_y=gaussian_mean_y,
            gaussian_std=gaussian_std,
        )

        # Reset Attributes for VRPL
        self.task_type = TASK_TYPE.VRPL
        self.distribution_type = distribution_type
        self.max_route_length = max_route_length

        # Generation Function Dictionary
        self.generate_func_dict = {
            VRPL_TYPE.UNIFORM: self._generate_uniform,
            VRPL_TYPE.GAUSSIAN: self._generate_gaussian,
        }

    def _generate_max_route_length(self, depots: np.ndarray, points: np.ndarray) -> float:
        """Generate a loose route length limit."""
        if self.max_route_length is not None:
            return self.max_route_length
        dists_to_depot = np.linalg.norm(points - depots, axis=1)
        return float(2.0 * np.sum(dists_to_depot) + 1.0)

    def _generate_uniform(self) -> VRPLTask:
        # Generate demands and capacity
        demands, capacity = self._generate_demands_and_capacity()

        # Generate uniform random coordinates in [0, 1]
        coords = np.random.uniform(0.0, 1.0, size=(self.nodes_num + 1, 2))
        depots = coords[0]
        points = coords[1:]
        max_route_length = self._generate_max_route_length(depots=depots, points=points)

        # Create VRPL Instance from Data
        task_data = VRPLTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        task_data.from_data(
            depots=depots,
            points=points,
            demands=demands,
            capacity=capacity,
            max_route_length=max_route_length
        )
        return task_data

    def _generate_gaussian(self) -> VRPLTask:
        # Generate demands and capacity
        demands, capacity = self._generate_demands_and_capacity()

        # Generate coordinates from a Gaussian distribution
        coords = np.random.normal(
            loc=(self.gaussian_mean_x, self.gaussian_mean_y),
            scale=self.gaussian_std,
            size=(self.nodes_num + 1, 2),
        )
        depots = coords[0]
        points = coords[1:]
        max_route_length = self._generate_max_route_length(depots=depots, points=points)

        # Create VRPL Instance from Data
        task_data = VRPLTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        task_data.from_data(
            depots=depots,
            points=points,
            demands=demands,
            capacity=capacity,
            max_route_length=max_route_length
        )
        return task_data
