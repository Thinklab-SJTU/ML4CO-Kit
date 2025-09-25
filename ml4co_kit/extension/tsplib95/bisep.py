# -*- coding: utf-8 -*-


class BiSep:
    def __init__(self, *, in_=None, out=' '):
        self.i = in_
        self.o = out

    def split(self, text, maxsplit=None):
        maxsplit = -1 if maxsplit is None else maxsplit
        return text.split(self.i, maxsplit=maxsplit)

    def join(self, items):
        o = ' ' if self.o is None else self.o
        return o.join(items)


def from_value(value):
    if value is None or isinstance(value, str):
        i, o = value, value
    else:
        try:
            i, o = value
        except Exception:
            raise ValueError('must be a string or an in/out '
                             f'tuple, not {value}')
    return BiSep(in_=i, out=o)
