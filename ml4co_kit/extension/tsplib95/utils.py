# -*- coding: utf-8 -*-
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


def nint(x):
    """Round a value to an integer.

    :param float x: original value
    :return: rounded integer
    :rtype: int
    """
    return int(x + 0.5)


def deltas(start, end):
    return (e - s for e, s in zip(end, start))


class RadianGeo:
    def __init__(self, coord):
        x, y = coord
        self.lat = self.__class__.parse_component(x)
        self.lng = self.__class__.parse_component(y)

    @staticmethod
    def parse_component(component):
        return math.radians(parse_degrees(component))


def _int_sum(n, memo={}):
    if n not in memo:
        s = n * (n + 1) // 2
        memo[n] = s
    return memo[n]


def integer_sum(n, m=None):
    s = _int_sum(n)
    if m:
        s -= _int_sum(m)
    return s


def pairwise(indexes):
    # double list double in case indexes is an iterator
    starts = list(indexes)
    ends = list(starts)

    # shift the ends by one, and make it circular
    ends += [ends.pop(0)]

    # pair up the neighbors
    return zip(starts, ends)


def friendly_join(items, limit=None):
    if not items:
        return ''

    if limit is not None:
        truncated = len(items) - limit
        items = items[:limit]
        if truncated > 0:
            items.append(f'{truncated} more')

    *items, last_item = items
    if not items:
        return str(last_item)

    # oxford commas are important
    if len(items) == 1:
        return f'{items[0]} and {last_item}'
    return f'{", ".join(items)}, and {last_item}'
