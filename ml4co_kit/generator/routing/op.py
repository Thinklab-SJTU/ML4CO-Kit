r"""
Generator for Orienteering Problem (OP) instances.
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
from ml4co_kit.task.routing.op import OPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class OP_TYPE(str, Enum):
    """Define the OP types as an enumeration."""
    UNIFORM = "uniform" # Uniform prizes
    CONSTANT = "constant" # Constant prizes
    DISTANCE = "distance" # Distance-based prizes


class OPGenerator(RoutingGeneratorBase):
    """Generator for Orienteering Problem (OP) instances."""
    
    def __init__(
        self, 
        distribution_type: OP_TYPE = OP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        max_length: float = 3.0, # nearly half of the TSP tour length
        # special args for uniform
        uniform_scale: tuple = (1, 100),
    ):
        # Super Initialization
        super(OPGenerator, self).__init__(
            task_type=TASK_TYPE.OP, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = nodes_num
        self.max_length = max_length
        
        # Special Args for Uniform
        self.uniform_scale = uniform_scale
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            OP_TYPE.UNIFORM: self._generate_uniform,
            OP_TYPE.CONSTANT: self._generate_constant,
            OP_TYPE.DISTANCE: self._generate_distance,
        }
    
    def _generate_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate demands and capacity"""
        coords = np.random.uniform(size=(self.nodes_num + 1, 2))
        depots = coords[0]
        points = coords[1:]
        return depots, points
    
    def _generate_uniform(self) -> OPTask:
        # Generate coordinates
        depots, points = self._generate_coords()
        
        # Generate prizes
        prizes: np.ndarray = np.random.randint(
            low=self.uniform_scale[0], 
            high=self.uniform_scale[1], 
            size=(self.nodes_num,)
        )
        prizes = prizes / self.uniform_scale[1]
        
        # Create OP Instance from Data
        data = OPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(
            depots=depots, points=points, prizes=prizes, 
            max_length=self.max_length
        )
        return data
    
    def _generate_constant(self) -> OPTask:
        # Generate coordinates
        depots, points = self._generate_coords()
        
        # Generate prizes
        prizes = np.ones(self.nodes_num)
        
        # Create OP Instance from Data
        data = OPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(
            depots=depots, points=points, prizes=prizes, 
            max_length=self.max_length
        )
        return data
    
    def _generate_distance(self) -> OPTask:
        # Generate coordinates
        depots, points = self._generate_coords()
        
        # Generate prizes
        dist2depot = np.linalg.norm(points - depots, axis=1)
        prizes = 0.01 + 0.99 * dist2depot / np.max(dist2depot)
        
        # Create OP Instance from Data
        data = OPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(
            depots=depots, points=points, prizes=prizes, 
            max_length=self.max_length
        )
        return data