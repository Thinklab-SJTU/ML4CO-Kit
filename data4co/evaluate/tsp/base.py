import math
import numpy as np
import scipy.spatial
from typing import Union

SUPPORT_TSPLIB_TYPE = ["EUC_2D", "GEO"]


class TSPEvaluator(object):
    def __init__(self, points: Union[list, np.ndarray], norm: str = "EUC_2D"):
        if type(points) == list:
            points = np.array(points)
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if points.ndim != 2:
            raise ValueError("points must be 2D array.")
        self.points = points
        self.set_norm(norm)
        self.edge_weights = self.get_weight()

    def set_norm(self, norm: str):
        if norm not in SUPPORT_TSPLIB_TYPE:
            message = (
                f"The norm type ({norm}) is not a valid type, "
                f"only {SUPPORT_TSPLIB_TYPE} are supported."
            )
            raise ValueError(message)
        self.norm = norm

    def get_weight(self):
        if self.norm == "EUC_2D":
            return scipy.spatial.distance_matrix(self.points, self.points)
        elif self.norm == "GEO":
            nodes_num = self.points.shape[0]
            edge_weights = np.zeros(shape=(nodes_num, nodes_num), dtype=np.int32)
            for i_idx in range(nodes_num):
                for j_idx in range(nodes_num):
                    if j_idx <= i_idx:
                        continue
                    node_coord_1 = self.points[i_idx]
                    node_coord_2 = self.points[j_idx]
                    weight = geographical(node_coord_1, node_coord_2)
                    edge_weights[i_idx][j_idx] = weight
                    edge_weights[j_idx][i_idx] = weight
            return edge_weights

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.edge_weights[route[i], route[i + 1]]
        return total_cost


def parse_degrees(coord):
    """Parse an encoded geocoordinate value into real degrees.

    :param float coord: encoded geocoordinate value
    :return: real degrees
    :rtype: float
    """
    degrees = int(coord)
    minutes = coord - degrees
    return degrees + minutes * 5 / 3


class RadianGeo:
    def __init__(self, coord):
        x, y = coord
        self.lat = self.__class__.parse_component(x)
        self.lng = self.__class__.parse_component(y)

    @staticmethod
    def parse_component(component):
        return math.radians(parse_degrees(component))


def geographical(start, end, radius=6378.388):
    """Return the geographical distance between start and end.

    This is capable of performing distance calculations for GEO problems.

    :param tuple start: *n*-dimensional coordinate
    :param tuple end: *n*-dimensional coordinate
    :param float radius: the radius of the Earth
    :return: rounded distance
    """
    if len(start) != len(end):
        raise ValueError("dimension mismatch between start and end")

    start = RadianGeo(start)
    end = RadianGeo(end)

    q1 = math.cos(start.lng - end.lng)
    q2 = math.cos(start.lat - end.lat)
    q3 = math.cos(start.lat + end.lat)
    distance = radius * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1

    return distance
