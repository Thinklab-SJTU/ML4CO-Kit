r"""
Generator for CVRP instances.
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
from typing import Union, Tuple
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class CVRP_TYPE(str, Enum):
    """Define the CVRP types as an enumeration."""
    UNIFORM = "uniform" # Uniform coords
    GAUSSIAN = "gaussian" # Gaussian coords


class CVRPGenerator(RoutingGeneratorBase):
    """Generator for Traveling Salesman Problem (CVRP) instances."""
    
    def __init__(
        self, 
        distribution_type: CVRP_TYPE = CVRP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for demands and capacity
        min_demand: int = 1,
        max_demand: int = 9,
        min_capacity: int = 40,
        max_capacity: int = 40,
        # special args for gaussian
        gaussian_mean_x: float = 0.0,
        gaussian_mean_y: float = 0.0,
        gaussian_std: float = 1.0,
    ):
        # Super Initialization
        super(CVRPGenerator, self).__init__(
            task_type=TASK_TYPE.CVRP, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = nodes_num
        
        # Special Args for Demands and Capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        
        # Special Args for Gaussian
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            CVRP_TYPE.UNIFORM: self._generate_uniform,
            CVRP_TYPE.GAUSSIAN: self._generate_gaussian,
        }
    
    def _generate_demands_and_capacity(self) -> Tuple[np.ndarray, int]:
        """Generate demands and capacity"""
        demands = np.random.randint(
            self.min_demand, self.max_demand+1, size=(self.nodes_num,)
        )
        capacity = np.random.randint(self.min_capacity, self.max_capacity+1)
        return demands, capacity
    
    def _generate_uniform(self) -> CVRPTask:
        # Generate demands and capacity
        demands, capacity = self._generate_demands_and_capacity()
        
        # Generate uniform random coordinates in [0, 1]
        coords = np.random.uniform(0.0, 1.0, size=(self.nodes_num + 1, 2))
        depots = coords[0]
        points = coords[1:]
        
        # Create CVRP Instance from Data
        data = CVRPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(
            depots=depots, points=points, demands=demands, capacity=capacity
        )
        return data

    def _generate_gaussian(self) -> CVRPTask:
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
        
        # Create CVRP Instance from Data
        data = CVRPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(
            depots=depots, points=points, demands=demands, capacity=capacity
        )
        return data