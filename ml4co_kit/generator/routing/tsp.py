r"""
Generator for TSP instances.
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
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class TSP_TYPE(str, Enum):
    """Define the TSP types as an enumeration."""
    UNIFORM = "uniform" # Uniform coords
    GAUSSIAN = "gaussian" # Gaussian coords
    CLUSTER = "cluster" # Cluster coords
    

class TSPGenerator(RoutingGeneratorBase):
    """Generator for Traveling Salesman Problem (TSP) instances."""
    
    def __init__(
        self, 
        distribution_type: TSP_TYPE = TSP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for gaussian
        gaussian_mean_x: float = 0.0,
        gaussian_mean_y: float = 0.0,
        gaussian_std: float = 1.0,
        # special args for cluster
        cluster_nums: int = 10,
        cluster_std: float = 0.1,
    ):
        # Super Initialization
        super(TSPGenerator, self).__init__(
            task_type=TASK_TYPE.TSP, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = nodes_num
        
        # Special Args for Gaussian
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std

        # Special Args for Cluster
        self.cluster_nums = cluster_nums
        self.cluster_std = cluster_std
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            TSP_TYPE.UNIFORM: self._generate_uniform,
            TSP_TYPE.GAUSSIAN: self._generate_gaussian,
            TSP_TYPE.CLUSTER: self._generate_cluster,
        }
        
    def _generate_uniform(self) -> TSPTask:
        # Generate uniform random coordinates in [0, 1]
        coords = np.random.uniform(0.0, 1.0, size=(self.nodes_num, 2))

        # Create TSP Instance from Data
        data = TSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(points=coords)
        return data

    def _generate_gaussian(self) -> TSPTask:
        # Generate coordinates from a Gaussian distribution
        coords = np.random.normal(
            loc=(self.gaussian_mean_x, self.gaussian_mean_y),
            scale=self.gaussian_std,
            size=(self.nodes_num, 2),
        )
        
        # Create TSP Instance from Data
        data = TSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(points=coords)
        return data

    def _generate_cluster(self) -> TSPTask:
        # Ensure cluster_nums is less than or equal to nodes_num
        if self.cluster_nums > self.nodes_num:
            raise ValueError(
                "Number of clusters must be less than or equal to number of nodes."
            )
        
        # Ensure nodes_num is divisible by cluster_nums
        if self.nodes_num % self.cluster_nums != 0:
            raise ValueError("Number of nodes must be divisible by number of clusters.")
        points_per_cluster = self.nodes_num // self.cluster_nums
        
        # Generate cluster centers and points around them
        cluster_centers = np.random.uniform(0, 1, size=(self.cluster_nums, 2))
        cluster_points = []
        for center in cluster_centers:
            points = np.random.normal(
                loc=center,
                scale=self.cluster_std,
                size=(points_per_cluster, 2),
            )
            cluster_points.append(points)
            
        # Return the coordinates of all points
        cluster_centers = np.vstack(cluster_points)
        coords = cluster_centers.astype(self.precision)

        # Create TSP Instance from Data
        data = TSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(points=coords)
        return data