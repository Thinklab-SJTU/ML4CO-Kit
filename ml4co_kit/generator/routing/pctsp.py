r"""
Generator for PCTSP instances.
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
from ml4co_kit.task.routing.pctsp import PCTSPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class PCTSP_TYPE(str, Enum):
    """Define the PCTSP types as an enumeration."""
    UNIFORM = "uniform" # Uniform prizes


class PCTSPGenerator(RoutingGeneratorBase):
    """Generator for Prize Collecting Traveling Salesman Problem (PCTSP) instances."""
    
    def __init__(
        self, 
        distribution_type: PCTSP_TYPE = PCTSP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for uniform
        uniform_k: float = 3.0 # nearly half of the TSP tour length
    ):
        # Super Initialization
        super(PCTSPGenerator, self).__init__(
            task_type=TASK_TYPE.PCTSP, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = nodes_num
        
        # Special Args for Uniform
        self.uniform_k = uniform_k
                
        # Generation Function Dictionary
        self.generate_func_dict = {
            PCTSP_TYPE.UNIFORM: self._generate_uniform,
        }
        
    def _generate_uniform(self) -> PCTSPTask:
        # Generate depots and points
        coords = np.random.uniform(size=(self.nodes_num + 1, 2))
        depots = coords[0]
        points = coords[1:]
        
        # Generate prizes
        prizes = np.random.uniform(size=(self.nodes_num,))
        prizes = prizes * 4 / self.nodes_num
        
        # Generate penalties
        penalties = np.random.uniform(size=(self.nodes_num,))
        penalties = penalties * 3 * self.uniform_k / self.nodes_num
        
        # Create PCTSP Instance from Data
        data = PCTSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(
            depots=depots, points=points, prizes=prizes, 
            penalties=penalties, required_prize=1.0
        )
        return data