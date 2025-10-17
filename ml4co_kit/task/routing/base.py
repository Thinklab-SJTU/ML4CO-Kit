r"""
Base Task Class for Routing Problems.
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


import math
import numpy as np
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class DISTANCE_TYPE(str, Enum):
    """Define the distance types as an enumeration."""

    # Euclidean Distance
    EUC_2D = "EUC_2D" # Euclidean Distance in 2D
    EUC_3D = "EUC_3D" # Euclidean Distance in 3D
    
    # Maximum Distance
    MAX_2D = "MAX_2D" # Maximum Distance in 2D
    MAX_3D = "MAX_3D" # Maximum Distance in 3D
    
    # Manhattan Distance
    MAN_2D = "MAN_2D" # Manhattan Distance in 2D
    MAN_3D = "MAN_3D" # Manhattan Distance in 3D
    
    # Geographical Distance
    GEO = "GEO"       # Geographical Distance
    
    # Att Distance
    ATT = "ATT"       # Att Distance


class ROUND_TYPE(str, Enum):
    """Define the rounding types as an enumeration."""
    NO = "no"         # No Rounding
    CEIL = "ceil"     # Ceiling
    FLOOR = "floor"   # Floor
    ROUND = "round"   # Round to Nearest Integer


class DisntanceEvaluator(object):
    """Distance evaluator for different distance types."""
    def __init__(
        self,
        distance_type: DISTANCE_TYPE,
        round_type: ROUND_TYPE,
        geo_radius: float = 6371.393
    ):
        # Initialize Attributes
        self.distance_type = distance_type
        self.round_type = round_type
        self.geo_radius = geo_radius
    
    @staticmethod
    def euclidean(start: np.ndarray, end: np.ndarray):
        """Return the Euclidean distance between start and end points."""
        deltas = (e - s for e, s in zip(end, start))
        square_distance = sum(d * d for d in deltas)
        distance = math.sqrt(square_distance)
        return distance
    
    @staticmethod
    def maximum(start: np.ndarray, end: np.ndarray):
        """Return the Maximum distance between start and end points."""
        deltas = (e - s for e, s in zip(end, start))
        distance = max(abs(d) for d in deltas)
        return distance
    
    @staticmethod
    def manhattan(start: np.ndarray, end: np.ndarray):
        """Return the Manhattan distance between start and end points."""
        deltas = (e - s for e, s in zip(end, start))
        distance = sum(abs(d) for d in deltas)
        return distance
    
    @staticmethod
    def att(start: np.ndarray, end: np.ndarray):
        """Return the Att distance between start and end points."""
        deltas = (e - s for e, s in zip(end, start))
        distance = math.sqrt(sum(d * d for d in deltas) / 10)
        distance = math.ceil(distance)
        if distance < math.sqrt(sum(d * d for d in deltas)):
            distance += 1
        return distance
    
    def geographical(self, start: np.ndarray, end: np.ndarray):
        """Return the Geographical distance between start and end points."""
        # Parse Degrees and Minutes
        def parse_geo(coord):
            deg = int(coord)
            min = coord - deg
            return deg + min * 5.0 / 3.0
        
        # Convert latitude and longitude to radians
        lat1 = math.radians(parse_geo(start[0]))
        lng1 = math.radians(parse_geo(start[1]))
        lat2 = math.radians(parse_geo(end[0]))
        lng2 = math.radians(parse_geo(end[1]))
        
        # Compute the spherical distance
        q1 = math.cos(lng1 - lng2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)
        distance = self.geo_radius * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1
        return distance
    
    def round_result(self, distance: float) -> int:
        """Return the rounded distance based on the specified rounding type."""
        if self.round_type == ROUND_TYPE.NO:
            return distance
        elif self.round_type == ROUND_TYPE.CEIL:
            return math.ceil(distance)
        elif self.round_type == ROUND_TYPE.FLOOR:
            return math.floor(distance)
        elif self.round_type == ROUND_TYPE.ROUND:
            return round(distance)
        else:
            raise ValueError(f'Unsupported rounding type: {self.round_type}')
    
    def cal_distance(self, start: np.ndarray, end: np.ndarray) -> float:
        """Return the distance based on the specified distance type."""
        # Check Dimension
        if len(start) != len(end):
            raise ValueError('dimension mismatch between start and end')
        
        # Calculate Distance
        if self.distance_type == DISTANCE_TYPE.EUC_2D:
            distance = self.euclidean(start=start, end=end)
        elif self.distance_type == DISTANCE_TYPE.EUC_3D:
            distance = self.euclidean(start=start, end=end)
        elif self.distance_type == DISTANCE_TYPE.MAX_2D:
            distance = self.maximum(start=start, end=end)
        elif self.distance_type == DISTANCE_TYPE.MAX_3D:
            distance = self.maximum(start=start, end=end)
        elif self.distance_type == DISTANCE_TYPE.MAN_2D:
            distance = self.manhattan(start=start, end=end)
        elif self.distance_type == DISTANCE_TYPE.MAN_3D:
            distance = self.manhattan(start=start, end=end)
        elif self.distance_type == DISTANCE_TYPE.GEO:
            distance = self.geographical(start=start, end=end)
        elif self.distance_type == DISTANCE_TYPE.ATT:
            distance = self.att(start=start, end=end)
        else:
            raise ValueError(f'Unsupported distance type: {self.distance_type}')

        # Return Rounded Distance
        return self.round_result(distance=distance)


class RoutingTaskBase(TaskBase):
    r"""
    Base class for all routing problems in the ML4CO kit.
    """
    
    def __init__(
        self, 
        task_type: TASK_TYPE, 
        minimize: bool,
        distance_type: DISTANCE_TYPE, 
        round_type: ROUND_TYPE,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(RoutingTaskBase, self).__init__(
            task_type=task_type, 
            minimize=minimize,
            precision=precision
        )
        
        # Initialize Attributes
        self.distance_type = distance_type
        self.dist_eval = DisntanceEvaluator(
            distance_type=distance_type, round_type=round_type
        )