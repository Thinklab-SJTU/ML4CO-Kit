r"""
Generator for SPCTSP instances.
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
from ml4co_kit.task.routing.spctsp import SPCTSPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class SPCTSP_TYPE(str, Enum):
    """Define the SPCTSP types as an enumeration."""
    UNIFORM = "uniform" # Uniform prizes


class SPCTSPGenerator(RoutingGeneratorBase):
    """Generator for SPCTSP instances."""
    
    def __init__(
        self, 
        distribution_type: SPCTSP_TYPE = SPCTSP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for uniform
        uniform_k: float = 3.0 # nearly half of the TSP tour length
    ):
        # Super Initialization
        super(SPCTSPGenerator, self).__init__(
            task_type=TASK_TYPE.SPCTSP, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = nodes_num
        
        # Special Args for Uniform
        self.uniform_k = uniform_k
                
        # Generation Function Dictionary
        self.generate_func_dict = {
            SPCTSP_TYPE.UNIFORM: self._generate_uniform,
        }
        
    def _generate_uniform(self) -> SPCTSPTask:
        # Generate depots and points
        coords = np.random.uniform(size=(self.nodes_num + 1, 2))
        depots = coords[0]
        points = coords[1:]
        
        # Generate prizes
        expected_prizes = np.random.uniform(size=(self.nodes_num,))
        expected_prizes = expected_prizes * 4 / self.nodes_num
        noise_factor = 2 * np.random.uniform(size=(self.nodes_num,))
        actual_prizes = noise_factor * expected_prizes
        
        # Generate penalties
        penalties = np.random.uniform(size=(self.nodes_num,))
        penalties = penalties * 3 * self.uniform_k / self.nodes_num
        
        # Create SPCTSP Instance from Data
        data = SPCTSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(
            depots=depots, 
            points=points, 
            expected_prizes=expected_prizes, 
            actual_prizes=actual_prizes, 
            penalties=penalties, 
            required_prize=1.0
        )
        return data