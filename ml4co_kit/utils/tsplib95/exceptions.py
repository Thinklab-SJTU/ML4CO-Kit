# -*- coding: utf-8 -*-


__all__ = [
    'TsplibError',
    'ParsingError',
    'RenderingError',
    'ValidationError',
]


class TsplibError(Exception):
    """Base exception for all tsplib95 errors."""

    @classmethod
    def wrap(cls, exc, message):
        if exc.args and exc.args[0]:
            message = f'{message}: {exc.args[0]}'
        return cls(message, *exc.args[1:])

    def amend(self, message):
        return self.__class__.wrap(self, message)


class ValidationError(TsplibError):
    """Exception raised when a problem fails validation."""


class ParsingError(TsplibError, ValueError):
    """Exception raised when a value cannot be parsed from the text."""


class RenderingError(TsplibError, ValueError):
    """Exception raised when a value cannot be rendered into text."""
