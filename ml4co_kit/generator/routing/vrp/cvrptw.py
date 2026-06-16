r"""
Generator for CVRPTW instances.
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
from ml4co_kit.task.routing.vrp.cvrptw import CVRPTWTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE
from ml4co_kit.generator.routing.vrp.cvrp import (
    CVRPGenerator, CVRP_TYPE, generate_for_tw
)


class CVRPTWGenerator(CVRPGenerator):
    """Generator for CVRPTW instances."""
    
    def __init__(
        self, 
        cvrp_open: bool = False,
        distribution_type: CVRP_TYPE = CVRP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for demands, capacity
        min_demand: int = 1,
        max_demand: int = 9,
        min_capacity: int = 40,
        max_capacity: int = 40,
        # special args for gaussian
        gaussian_mean_x: float = 0.0,
        gaussian_mean_y: float = 0.0,
        gaussian_std: float = 1.0,
        # special args for TW
        max_time: float = 4.6,
    ):
        # Super Initialization
        super(CVRPTWGenerator, self).__init__(
            cvrp_open=cvrp_open,
            distribution_type=distribution_type, 
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
        
        # Set Task Type
        self.task_type = TASK_TYPE.CVRPTW

        # Extra Attributes
        self.max_time = max_time

    def _generate_core(
        self, depots: np.ndarray, points: np.ndarray
    ) -> CVRPTWTask:
        # Generate demands and capacity
        demands, capacity = self._generate_demands_and_capacity()

        # Generate the time window and service time
        d0i = np.linalg.norm(depots - points, axis=1)
        tw, service = generate_for_tw(d0i, self.max_time)
        
        # Create CVRPTW Instance from Data
        task_data = CVRPTWTask(
            cvrp_open=self.cvrp_open,
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        task_data.from_data(
            depots=depots, points=points, demands=demands, 
            capacity=capacity, tw=tw, service=service
        )
        return task_data