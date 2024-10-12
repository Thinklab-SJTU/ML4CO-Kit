import math


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