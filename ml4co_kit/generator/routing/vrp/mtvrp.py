r"""
Generator for MTVRP instances.
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
from ml4co_kit.task.routing.vrp.mtvrp import MTVRPTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE
from ml4co_kit.generator.routing.vrp.cvrp import (
    CVRPGenerator, CVRP_TYPE, generate_for_tw, generate_for_l
)


class MTVRPGenerator(CVRPGenerator):
    """Generator for MTVRP instances."""
    
    def __init__(
        self, 
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
        # special args for B-L-TW
        bh_ratio: float = 0.2,
        rou_max: float = 2.8,
        max_time: float = 4.6,
        # special args for flags
        open_ratio: float = 0.5,
        backhaul_ratio: float = 0.5,
        mixed_backhaul_ratio: float = 0.5,
        tw_ratio: float = 0.5,
        max_route_length_ratio: float = 0.5,
    ):
        # Super Initialization
        super(MTVRPGenerator, self).__init__(
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
        self.task_type = TASK_TYPE.MTVRP

        # Extra Attributes
        self.bh_ratio = bh_ratio
        self.rou_max = rou_max
        self.max_time = max_time

        # Ratio for each task type
        self.open_ratio = open_ratio
        self.backhaul_ratio = backhaul_ratio
        self.mixed_backhaul_ratio = mixed_backhaul_ratio
        self.tw_ratio = tw_ratio
        self.max_route_length_ratio = max_route_length_ratio

    def _generate_core(
        self, depots: np.ndarray, points: np.ndarray
    ) -> MTVRPTask:
        # Randomly get flags
        flag_o = np.random.rand() < self.open_ratio
        flag_b = np.random.rand() < self.backhaul_ratio
        flag_mb = np.random.rand() < self.mixed_backhaul_ratio
        flag_tw = np.random.rand() < self.tw_ratio
        flag_l = np.random.rand() < self.max_route_length_ratio
        flag_mb = False if flag_b == False else flag_mb
        
        # Generate data related to C and B
        demands, capacity = self._generate_demands_and_capacity()
        if flag_b:
            bh_mask = np.random.rand(self.nodes_num) < self.bh_ratio
            demands[bh_mask] = -demands[bh_mask]

        # Generate data related to TW and L
        if flag_tw or flag_l:
            d0i = np.linalg.norm(depots - points, axis=1)
            if flag_tw:
                tw, service = generate_for_tw(d0i, self.max_time)
            else:
                tw, service = None, None
            if flag_l:
                max_route_length = generate_for_l(d0i, self.rou_max)
            else:
                max_route_length = None
        else:
            tw, service, max_route_length = None, None, None

        # Create MTVRP Instance from Data
        task_data = MTVRPTask(
            cvrp_open=flag_o,
            backhaul_flag=flag_b,
            mixed_backhaul=flag_mb,
            tw_flag=flag_tw,
            max_route_length_flag=flag_l,
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )

        # Set Data to MTVRP Instance
        task_data.from_data(
            depots=depots, points=points, demands=demands, 
            capacity=capacity, tw=tw, service=service, 
            max_route_length=max_route_length
        )
        return task_data