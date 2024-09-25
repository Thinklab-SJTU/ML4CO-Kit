# -*- coding: utf-8 -*-
import re

from . import transformers as T
from . import exceptions


__all__ = [
    'Field',
    'TransformerField',
    'StringField',
    'IntegerField',
    'FloatField',
    'NumberField',
    'IndexedCoordinatesField',
    'AdjacencyListField',
    'EdgeListField',
    'MatrixField',
    'EdgeDataField',
    'DepotsField',
    'DemandsField',
    'ToursField',
]


class Field:
    """Contains base functionality for all fields.

    The default value can be a callable, in which case it is invoked for each
    call to :func:`get_default_value`. The default can be set on an instance
    or as a class attribute, but the class attribute is only checked when the
    field is initially created.

    :param str keyword: keyword (typically all caps)
    :param default: a default value or callable that will return a default
    """

    default = None

    def __init__(self, keyword, *, default=None):
        self.keyword = keyword
        self.default = default or self.__class__.default

    def __repr__(self):
        return f'<{self.__class__.__qualname__}({repr(self.keyword)})>'

    def get_default_value(self):
        """Return the default value.

        Callables are called for a default value to return each time.

        :return: the default value
        :rtype: Any
        """
        if callable(self.default):
            return self.default()
        return self.default

    def parse(self, text):
        """Convert text into a value.

        This must be implemented in a subclass.

        :param str text:
        :return: a value
        """
        raise NotImplementedError()

    def render(self, value):
        """Convert a value into text.

        This must be implemented in a subclass.

        :param value: a value
        :return: text
        :rtype: str
        """
        raise NotImplementedError()

    def validate(self, value):
        """Validate a value.

        Raise an exception if the value fails validation.

        The default implementation does nothing.

        :param value: a value
        :raises Exception: if the value does not pass validation
        """


class TransformerField(Field):
    """Field that delegates to a :class:`~tsplib95.transformers.Transformer`.

    :param str keyword: keyword
    :param callable transformer: transformer to use
    """

    def __init__(self, keyword, *, transformer=None, **kwargs):
        super().__init__(keyword, **kwargs)
        self.tf = transformer

    @classmethod
    def build_transformer(cls):
        """Construct an appropriate transformer for the field.

        :return: transformer
        :rtype: :class:`~tsplib95.transformers.Transformer`
        """
        return T.Transformer()

    @property
    def tf(self):
        if self._tf is None:
            self._tf = self.build_transformer()
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = value

    def parse(self, text):
        """Parse the text into a value using the transformer.

        :param str text: text to parse
        :return: value
        """
        try:
            return self.tf.parse(text)
        except exceptions.ParsingError as e:
            context = f'{self.__class__.__qualname__}({self.keyword})'
            raise exceptions.ParsingError.wrap(e, context)

    def render(self, value):
        """Render the value into text using the transformer.

        :param str text: value to render
        :return: text
        """
        try:
            return self.tf.render(value)
        except exceptions.RenderingError as e:
            context = f'{self.__class__.__qualname__}({self.keyword})'
            raise exceptions.RenderingError.wrap(e, context)

    def validate(self, value):
        """Validate the value using the transformer.

        :param str text: value to validate
        """
        return self.tf.validate(value)


class StringField(TransformerField):
    """Simple string field."""

    @classmethod
    def build_transformer(cls):
        return T.FuncT(func=str)


class IntegerField(TransformerField):
    """Simple integer field."""

    default = 0

    @classmethod
    def build_transformer(cls):
        return T.FuncT(func=int)


class FloatField(TransformerField):
    """Simple float field."""

    default = 0.0

    @classmethod
    def build_transformer(cls):
        return T.FuncT(func=float)


class NumberField(TransformerField):
    """Number field, supporting ints and floats."""

    default = 0

    @classmethod
    def build_transformer(cls):
        return T.NumberT()


class IndexedCoordinatesField(TransformerField):
    """Field for coordinates by index.

    When given, ``dimensions`` stipulates the possible valid dimensionalities
    for the coordinates. For exapmle, ``dimensions=(2, 3)`` indicates the
    coordinates are either all 2d or all 3d, whereas ``dimensions=2`` indicates
    all coordinates must be 2d. The check is only enforced during validation.

    :param dimensions: one or more valid dimensionalities
    """

    default = dict

    def __init__(self, *args, dimensions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimensions = self._tuplize(dimensions)

    @staticmethod
    def _tuplize(dimensions):
        # helper to accept either a tuple, a single value, or None
        try:
            return tuple(iter(dimensions))
        except Exception:
            return (dimensions,) if dimensions else None

    @classmethod
    def build_transformer(cls):
        key = T.FuncT(func=int)
        value = T.ListT(value=T.NumberT())
        return T.MapT(key=key, value=value, sep='\n')

    def validate(self, value):
        super().validate(value)
        cards = set(len(coord) for coord in value.values())
        if self.dimensions and cards not in ({dim} for dim in self.dimensions):
            error = ('all coordinates must have the same dimensionality '
                     f'and it must be one of {self.dimensions}')
            raise exceptions.ValidationError(error)


class AdjacencyListField(TransformerField):
    """Field for an adjancency list."""

    default = dict

    @classmethod
    def build_transformer(cls):
        return T.MapT(key=T.FuncT(func=int),
                      value=T.ListT(value=T.FuncT(func=int)),
                      sep=('-1', ' -1\n'),
                      terminal='-1')


class EdgeListField(TransformerField):
    """Field for a list of edges."""

    default = list

    @classmethod
    def build_transformer(cls):
        edge = T.ListT(value=T.FuncT(func=int), size=2)
        return T.ListT(value=edge, terminal='-1', sep='\n')


class MatrixField(TransformerField):
    """Field for a matrix of numbers."""

    default = list

    @classmethod
    def build_transformer(cls):
        row = T.ListT(value=T.NumberT())
        return T.ListT(value=row, sep='\n')


class EdgeDataField(TransformerField):
    """Field for edge data."""

    default = dict

    @classmethod
    def build_transformer(cls):
        adj_list = AdjacencyListField.build_transformer()
        edge_list = EdgeListField.build_transformer()
        return T.UnionT(adj_list, edge_list)


class DepotsField(TransformerField):
    """Field for depots."""

    default = list

    @classmethod
    def build_transformer(cls):
        depot = T.FuncT(func=int)
        return T.ListT(value=depot, terminal='-1')


class DemandsField(TransformerField):
    """Field for demands."""

    default = dict

    @classmethod
    def build_transformer(cls):
        node = T.FuncT(func=int)
        demand = T.FuncT(func=int)
        return T.MapT(key=node, value=demand, sep='\n')


class ToursField(Field):
    """Field for one or more tours."""

    default = list

    def __init__(self, *args, require_terminal=True):
        super().__init__(*args)
        self.terminal = '-1'
        self.require_terminal = require_terminal
        self._end_terminals = re.compile(rf'(?:(?:\s+|\b|^){self.terminal})+$')
        self._any_terminal = re.compile(rf'(?:\s+|\b){self.terminal}(?:\b|\s+)')  # noqa: E501

    def parse(self, text):
        """Parse the text into a list of tours.

        :param str text: text to parse
        :return: tours
        :rtype: list
        """
        text = text.strip()
        if not text:
            return []

        match = self._end_terminals.search(text)
        # terminal must terminate, if required
        if not match and self.require_terminal:
            terminal = text.split()[-1]
            error = (f'must terminate in "{self.terminal}", '
                     f'not {repr(terminal)}')
            raise exceptions.ParsingError(error)

        # trim the terminal if present
        if match:
            text = text[:match.start()]

        # split the texts and filter out the empties
        texts = self._any_terminal.split(text)
        texts = list(filter(None, texts))
        if not texts:
            return []

        # convert the tours from texts to integer lists
        tours = []
        for text in texts:
            try:
                tour = [int(n) for n in text.strip().split()]
            except ValueError as e:
                error = f'could not convert text to node index: {repr(e)}'
                raise exceptions.ParsingError(error)
            else:
                tours.append(tour)

        return tours

    def render(self, tours):
        """Render the tours as text.

        :param list tours: tours to render
        :return: rendered text
        :rtype: str
        """
        if not tours:
            return ''

        tour_strings = []
        for tour in tours:
            if tour:
                tour_string = ' '.join(str(i) for i in tour)
                tour_strings.append(f'{tour_string} -1')

        if tour_strings:
            tour_strings += ['-1']

        return '\n'.join(tour_strings)
