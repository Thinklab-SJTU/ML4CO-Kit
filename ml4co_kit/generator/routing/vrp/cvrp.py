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
from ml4co_kit.task.routing.vrp.cvrp import CVRPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class CVRP_TYPE(str, Enum):
    """Define the CVRP types as an enumeration."""
    UNIFORM = "uniform" # Uniform coords
    GAUSSIAN = "gaussian" # Gaussian coords


class CVRPGenerator(RoutingGeneratorBase):
    """Generator for CVRP instances."""
    
    def __init__(
        self, 
        cvrp_open: bool = False,
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
        self.cvrp_open = cvrp_open
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
        """
        Generate demands and capacity

        @article{
            nazari2018reinforcement,
            title={Reinforcement learning for solving the vehicle routing problem},
            author={Nazari, Mohammadreza and Oroojlooy, Afshin and Snyder, Lawrence and Tak{\'a}c, Martin},
            journal={Advances in neural information processing systems},
            volume={31},
            year={2018}
        }
        """
        demands = np.random.randint(
            self.min_demand, self.max_demand+1, size=(self.nodes_num,)
        )
        capacity = np.random.randint(self.min_capacity, self.max_capacity+1)
        return demands, capacity

    def _generate_core(self, depots: np.ndarray, points: np.ndarray) -> CVRPTask:
        # Generate demands and capacity
        demands, capacity = self._generate_demands_and_capacity()
        
        # Create CVRP Instance from Data
        task_data = CVRPTask(
            cvrp_open=self.cvrp_open,
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        task_data.from_data(
            depots=depots, points=points, 
            demands=demands, capacity=capacity
        )
        return task_data

    def _generate_uniform(self):
        # Generate uniform random coordinates in [0, 1]
        coords = np.random.uniform(0.0, 1.0, size=(self.nodes_num + 1, 2))
        depots = coords[0]
        points = coords[1:]

        # Create CVRPBLTW Instance from Data
        return self._generate_core(depots, points)

    def _generate_gaussian(self):
        # Generate coordinates from a Gaussian distribution
        coords = np.random.normal(
            loc=(self.gaussian_mean_x, self.gaussian_mean_y),
            scale=self.gaussian_std,
            size=(self.nodes_num + 1, 2),
        )
        depots = coords[0]
        points = coords[1:]

        # Create CVRPBLTW Instance from Data
        return self._generate_core(depots, points)


######################################
#  Generate Utils for CVRP Variants  #
######################################

def generate_for_l(
    d0i: np.ndarray, 
    upper_bound: float = 2.8  # 2sqrt(2) ~= 2.8
) -> float:
    """
    Generate the route length limit.
    Reference: https://github.com/ai4co/routefinder
    """
    # Get the lower bound and upper bound
    lower_bound = 2 * np.max(d0i) + 1e-6
    upper_bound = max(lower_bound + 1e-6, upper_bound)

    # Sample the route length limit from U(lower_bound, upper_bound)
    # Where lower_bound is 2 * max_dist + 1e-6
    route_length = np.random.uniform(lower_bound, upper_bound)
    return route_length


def generate_for_tw(
    d0i: np.ndarray,
    max_time: float = 4.6,
    service_range: tuple = (0.15, 0.18),
    tw_range: tuple = (0.18, 0.2),
):
    """
    Generate the service time and time window.
    Reference: https://github.com/ai4co/routefinder
    Note: We set the speed of each vehicle to 1.0 for simplicity.
    """
    # Generate the service duration and 
    # time window length for each customer
    nodes_num = d0i.shape[0]
    tw_length = np.random.uniform(
        low=tw_range[0], 
        high=tw_range[1],
        size=(nodes_num,)
    )
    service = np.random.uniform(
        low=service_range[0], 
        high=service_range[1],
        size=(nodes_num,)
    )

    # Generate time window
    upper_bound = (max_time - service - tw_length) / d0i - 1
    tw_start = (
        (1 + (upper_bound - 1) * np.random.rand(nodes_num)) * d0i
    )
    tw_end = tw_start + tw_length
    tw = np.stack([tw_start, tw_end], axis=-1)
    
    # Add depot's time window and service time
    tw = np.concatenate(
        [np.array([[0.0, max_time]]), tw], axis=0
    )
    service = np.concatenate([np.array([0.0]), service], axis=0)
    return tw, service