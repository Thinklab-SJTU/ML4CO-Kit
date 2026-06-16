r"""
Generator for CVRPB instances.
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
from ml4co_kit.task.routing.vrp.cvrpb import CVRPBTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE
from ml4co_kit.generator.routing.vrp.cvrp import CVRPGenerator, CVRP_TYPE


class CVRPBGenerator(CVRPGenerator):
    """Generator for CVRPB instances."""
    
    def __init__(
        self, 
        cvrp_open: bool = False,
        mixed_backhaul: bool = False,
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
        # special args for B
        bh_ratio: float = 0.2,
    ):
        # Super Initialization
        super(CVRPBGenerator, self).__init__(
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
        
        # Set Task Type and Mixed Backhaul
        self.task_type = TASK_TYPE.CVRPB
        self.mixed_backhaul = mixed_backhaul

        # Extra Attributes
        self.bh_ratio = bh_ratio

    def _generate_core(
        self, depots: np.ndarray, points: np.ndarray
    ) -> CVRPBTask:
        # Generate demands and capacity
        demands, capacity = self._generate_demands_and_capacity()
        
        # Backhauls
        bh_mask = np.random.rand(self.nodes_num) < self.bh_ratio
        demands[bh_mask] = -demands[bh_mask]

        # Create CVRPB Instance from Data
        task_data = CVRPBTask(
            cvrp_open=self.cvrp_open,
            mixed_backhaul=self.mixed_backhaul,
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        task_data.from_data(
            depots=depots, points=points, 
            demands=demands, capacity=capacity
        )
        return task_data