# -*- coding: utf-8 -*-
from . import bisep
from . import exceptions
from . import utils


__all__ = [
    'Transformer',
    'FuncT',
    'NumberT',
    'ContainerT',
    'ListT',
    'MapT',
    'UnionT',
]


class Transformer:
    """Reusable transformer between text and data."""

    def parse(self, text):
        """Return the value of the text.

        :param str text: the text
        :return: the value
        :raises ~tsplib95.exceptions.ParsingError: if the text cannot be parsed
                                                   into a value
        """
        return text

    def render(self, value):
        """Return the text for the value.

        :param str value: the value
        :return: the text
        """
        if value is None:
            return ''
        return str(value)

    def validate(self, value):
        """Validate the value.

        :param value: the value
        """


class FuncT(Transformer):
    """Transformer that simply wraps a parsing function.

    The parsing function must accept a single positional argument for the text
    to parse. It must parse and return the value.

    Values are rendered back into values using the builtin :func:`str`, so it's
    generally best to use this for primitives.

    :param callable func: parsing function
    """

    def __init__(self, *, func):
        super().__init__()
        self.func = func

    def parse(self, text):
        try:
            return self.func(text)
        except Exception as e:
            raise exceptions.ParsingError.wrap(e, 'func transformer error')


class NumberT(Transformer):
    """Transformer for any number, int or float."""

    def parse(self, text):
        for func in (int, float):
            try:
                return func(text)
            except ValueError:
                pass
        error = f'could not convert text to number: {text}'
        raise exceptions.ParsingError(error)


class ContainerT(Transformer):
    """Transformer that acts as a generic container.

    :param value: transformer for each item
    :type value: :class:`~tsplib95.transformers.Transformer`
    :param str sep: separator between items
    :param str terminal: text that marks the end
    :param bool terminal_required: whether the terminal is required
    :param int size: required number of items
    :param bool filter_empty: filter out empty items (zero-length/blank)
    """

    def __init__(self, *, value=None, sep=None, terminal=None,
                 terminal_required=True, size=None, filter_empty=True):
        self.child_tf = value or Transformer()
        self.sep = bisep.from_value(sep)
        self.terminal = terminal
        self.terminal_required = terminal_required
        self.size = size
        self.filter_empty = filter_empty

    def parse(self, text):
        """Parse the text into a container of items.

        :param str text: the text to parse
        :return: container
        """
        # start without unpredictable whitespace
        text = text.strip()

        # if we have a terminal, make sure it's there and remove it
        if self.terminal:
            if not text.endswith(self.terminal) and self.terminal_required:
                raise exceptions.ParsingError(f'must end with {self.terminal}, '  # noqa: E501
                                              f'not "{text[-len(self.terminal):]}"')  # noqa: E501
            text = text[:-len(self.terminal)].strip()

        # split the whole text into multiple texts
        try:
            texts = self.split_items(text)
        except Exception as e:
            context = 'could not split the text'
            raise exceptions.ParsingError.wrap(e, context)

        # filter out the empties, or don't; it's configurable
        if self.filter_empty:
            texts = list(filter(None, texts))

        # if specified, make sure there's no terminal somwhere in the middle
        if self.terminal is not None:
            try:
                index = texts.index(self.terminal)
            except ValueError:
                pass  # happy path
            else:
                # uh-oh, found one, so raise an exception with the extra items
                extra = texts[index + 1:]
                error = (f'found {len(extra)} extra items after terminal '
                         f'{repr(self.terminal)}, first is {repr(extra[0])}')
                raise exceptions.ParsingError(error)

        # parse the texts into items, catching all errors by index
        items = []
        errors = []
        for i, text in enumerate(texts):
            try:
                item = self.parse_item(text)
            except Exception as e:
                errors.append(f'item.{i}=>{repr(e)}')
            else:
                items.append(item)

        # join and report any errors
        if errors:
            error = utils.friendly_join(errors, limit=3)
            raise exceptions.ParsingError(error)

        # if the size is specified, make sure its right
        if self.size and len(items) != self.size:
            error = f'expected {self.size} items, found {len(items)}'
            raise exceptions.ParsingError(error)

        # finally, pack the items into a container and return it
        return self.pack(items)

    def render(self, container):
        """Render the container into text.

        :param container: container to render
        :return: text
        """
        # unpack the items from the container and render them
        items = self.unpack(container)
        rendered = list(self.render_item(i) for i in items)

        # if there is a terminal, append it
        if self.terminal:
            rendered.append(self.terminal)

        # return the rendered items joined together
        return self.join_items(rendered)

    def parse_item(self, text):
        """Parse the text into a single item.

        :param str text: the text to parse
        :return: container
        """
        return self.child_tf.parse(text)

    def render_item(self, item):
        """Render the item into text.

        :param item: item to render
        :return: text
        """
        return self.child_tf.render(item)

    def split_items(self, text):
        """Split the text into multiple items.

        :param str text: text to split
        :return: muliple items
        :rtype: list
        """
        return self.sep.split(text)

    def join_items(self, items):
        """Join zero or more items into a single text.

        :param list items: items to join
        :return: joined text
        :rtype: str
        """
        return self.sep.join(items)

    def pack(self, items):
        """Pack the given items into a container.

        :param list items: items to pack
        :return: container with items in it
        """
        raise NotImplementedError()

    def unpack(self, container):
        """Unpack items from the container.

        :param container: container with items in it
        :return: items from container
        :rtype: list
        """
        raise NotImplementedError()


class ListT(ContainerT):
    """Transformer for a list of items."""

    def pack(self, items):
        return list(items)

    def unpack(self, container):
        return list(container)


class MapT(ContainerT):
    """Transformer for a key-value mapping of items.

    :param str kv_sep: separator between keys and values
    :param key: transformer for the keys
    :type key: :class:`~tsplib95.transformers.Transformer`
    """

    def __init__(self, *, kv_sep=None, key=None, **kwargs):
        super().__init__(**kwargs)
        self.key_tf = key or Transformer()
        self.kv_sep = bisep.from_value(kv_sep)

    def parse_item(self, text):
        # split the text into a key and value
        try:
            key, value = self.kv_sep.split(text, maxsplit=1)
        except ValueError:
            raise exceptions.ParsingError('expected key-value pair')

        # parse each and return a parsed key-value tuple
        return self.parse_key(key), self.parse_value(value)

    def render_item(self, item):
        # render the key and value
        k, v = item
        key = self.render_key(k)
        value = self.render_value(v)

        # return them joined
        return self.kv_sep.join([key, value])

    def parse_key(self, text):
        """Parse the text into a key.

        :param str text: the text to parse
        :return: key
        """
        return self.key_tf.parse(text)

    def render_key(self, key):
        """Render the key into text.

        :param key: the key to render
        :return: text
        :rtype: str
        """
        return self.key_tf.render(key)

    def parse_value(self, text):
        """Parse the text into a value.

        :param str text: the text to parse
        :return: value
        """
        return self.child_tf.parse(text)

    def render_value(self, value):
        """Render the value into text.

        :param value: the value to render
        :return: text
        :rtype: str
        """

        return self.child_tf.render(value)

    def pack(self, items):
        return {k: v for k, v in items}

    def unpack(self, container):
        return container.items()


class UnionT(Transformer):
    def __init__(self, *tfs, **kwargs):
        super().__init__(**kwargs)
        self.transformers = tfs

    def parse(self, text):
        errors = []
        for tf in self.transformers:
            try:
                return tf.parse(text)
            except Exception as e:
                errors.append(f'{tf.__class__.__qualname__}=>{repr(e)}')
        raise exceptions.ParsingError('no transformer in the union could '
                                      'parse the text, resulting in the '
                                      'following errors: '
                                      f'{utils.friendly_join(errors)}')

    def render(self, value):
        errors = []
        for tf in self.transformers:
            try:
                return tf.render(value)
            except Exception as e:
                errors.append(f'{tf.__class__.__qualname__}=>{repr(e)}')
        raise exceptions.RenderingError('no transformer in the union could '
                                        'render the value, resulting in the '
                                        'following errors: '
                                        f'{utils.friendly_join(errors)}')
