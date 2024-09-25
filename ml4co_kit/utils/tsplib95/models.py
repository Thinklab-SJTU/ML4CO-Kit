# -*- coding: utf-8 -*-
import itertools
import re

import networkx

from . import fields as F
from . import matrix
from . import distances
from . import utils


class FileMeta(type):
    def __new__(mcs, class_name, bases, attrs):
        # we need to map the fields by keyword and by field name
        # data is a dictionary of class attributes, some of which are fields
        # we need to pop the fields out and use them to create two maps
        # one by keyword and one by field name

        # first, we transform all Field attrs into these mappings
        current = {
            'fields_by_name': {},
            'fields_by_keyword': {},
            'names_by_keyword': {},
            'keywords_by_name': {},
        }
        for name, value in list(attrs.items()):
            if isinstance(value, F.Field):
                current['fields_by_name'][name] = value
                current['fields_by_keyword'][value.keyword] = value
                current['names_by_keyword'][value.keyword] = name
                current['keywords_by_name'][name] = value.keyword
                attrs.pop(name)
        attrs.update(current)

        # use the data to build a new class
        new_class = super().__new__(mcs, class_name, bases, attrs)

        for key in current:
            # merge together the data from all classes in the class
            # hierarchy by traversing them in reverse MRO order.
            data = {}
            for base in reversed(new_class.__mro__):
                if hasattr(base, key):
                    data.update(getattr(base, key))

                # Be sure to take care of attribute hiding. This
                # feature allows subclasses to "remove" (hide) fields
                # already defined by one of the parent classes.
                for name, base_value in list(base.__dict__.items()):
                    if base_value is None and name in data:
                        value = data.pop(name)
                        if isinstance(value, F.Field):
                            work = {
                                name: ['fields_by_name',
                                       'keywords_by_name'],
                                value.keyword: ['fields_by_keyword',
                                                'names_by_keyword'],
                            }
                            for k2, keys in work.items():
                                for k1 in keys:
                                    try:
                                        del data[k1][k2]
                                    except KeyError:
                                        pass  # TODO: logging

            # set the final value on the new class
            setattr(new_class, key, data)

        return new_class


class Problem(metaclass=FileMeta):
    """Base class for all problems.

    :param data: name-value data
    """

    def __init__(self, **data):
        super().__init__()
        # every keyword argument becomes an attribute
        for name, value in data.items():
            setattr(self, name, value)
        self._defaults = {}

    @classmethod
    def parse(cls, text, **options):
        """Parse text into a problem instance.

        Any keyword options are passed to the class constructor. If a keyword
        argument has the same name as a field then they will collide and cause
        an error.

        :param str text: problem text
        :param options: any keyword arguments to pass to the constructor
        :return: problem instance
        :rtype: :class:`Problem`
        """
        # prepare the regex for all known keys
        keywords = '|'.join(cls.fields_by_keyword)
        sep = r'''\s*:\s*|\s*\n'''
        pattern = f'({keywords}|EOF)(?:{sep})'

        # split the whole text by known keys
        regex = re.compile(pattern, re.M)
        __, *results = regex.split(text)

        # pair keys and values
        field_keywords = results[::2]
        field_values = results[1::2]

        # parse into a dictionary
        data = {}
        for keyword, value in zip(field_keywords, field_values):
            if keyword != 'EOF':
                field = cls.fields_by_keyword[keyword]
                name = cls.names_by_keyword[keyword]
                data[name] = field.parse(value.strip())

        # return as a model, letting options and field data potentially collide
        return cls(**data, **options)

    @classmethod
    def load(cls, filepath, **options):
        """Load a problem instance from a text file.

        Any keyword options are passed to the class constructor. If a keyword
        argument has the same name as a field then they will collide and cause
        an error.

        :param str filepath: path to a problem file
        :param options: any keyword arguments to pass to the constructor
        :return: problem instance
        :rtype: :class:`Problem`
        """
        with open(filepath) as f:
            return cls.read(f, **options)

    @classmethod
    def read(cls, fp, **options):
        """Read a problem instance from a file-like object.

        Any keyword options are passed to the class constructor. If a keyword
        argument has the same name as a field then they will collide and cause
        an error.

        :param str fp: a file-like object
        :param options: any keyword arguments to pass to the constructor
        :return: problem instance
        :rtype: :class:`Problem`
        """
        return cls.parse(fp.read(), **options)

    def __str__(self):
        return self.render()

    def __getattribute__(self, name):
        # check for a value like normal
        try:
            attrs = object.__getattribute__(self, '__dict__')
            return attrs[name]
        except KeyError:
            pass

        # value missing, so try to return the default
        # for the correpsonding field
        try:
            cls = object.__getattribute__(self, '__class__')
            field = cls.fields_by_name[name]
        except KeyError:
            pass
        else:
            # return a single default object
            default = field.get_default_value()
            if name in self._defaults and self._defaults[name] != default:
                # if the default has been altered, set it as the value
                setattr(self, name, self._defaults[name])
            else:
                # we don't have a default yet, save this one
                self._defaults[name] = default
            return self._defaults[name]

        # not a field, so punt to the super implementation
        return super().__getattribute__(name)

    def as_dict(self, by_keyword=False):
        """Return the problem data as a dictionary.

        :param bool by_keyword: use keywords (True) or names (False) or keys
        :return: problem data
        :rtype: dict
        """
        data = {}
        for name, field in self.__class__.fields_by_name.items():
            value = getattr(self, name)
            if name in self.__dict__ or value != field.get_default_value():
                key = field.keyword if by_keyword else name
                data[key] = value
        return data

    def as_name_dict(self):
        """Return the problem data as a dictionary by field name.

        :return: problem data
        :rtype: dict
        """
        return self.as_dict(by_keyword=False)

    def as_keyword_dict(self):
        """Return the problem data as a dictionary by field keyword.

        :return: problem data
        :rtype: dict
        """
        return self.as_dict(by_keyword=True)

    def render(self):
        # render each value by keyword
        rendered = self.as_name_dict()
        for name in list(rendered):
            value = rendered.pop(name)
            field = self.__class__.fields_by_name[name]
            if name in self.__dict__ or value != field.get_default_value():
                rendered[field.keyword] = field.render(value)

        # build keyword-value pairs with the separator
        kvpairs = []
        for keyword, value in rendered.items():
            sep = ':\n' if '\n' in value else ': '
            kvpairs.append(f'{keyword}{sep}{value}')
        kvpairs.append('EOF')

        # join and return the result
        return '\n'.join(kvpairs)

    def save(self, filename):
        with open(filename, 'w') as f:
            self.write(f)

    def write(self, fp):
        fp.write(self.render())

    def validate(self):
        pass


class StandardProblem(Problem):
    """Standard problem as outlined in the original TSLIB95 documentation.

    The available fields and their keywords are:

     * ``name`` - NAME
     * ``comment`` - COMMENT
     * ``type`` - TYPE
     * ``dimension`` - DIMENSION
     * ``capacity`` - CAPACITY
     * ``edge_weight_type`` - EDGE_WEIGHT_TYPE
     * ``edge_weight_format`` - EDGE_WEIGHT_FORMAT
     * ``edge_data_format`` - EDGE_DATA_FORMAT
     * ``node_coord_type`` - NODE_COORD_TYPE
     * ``display_data_type`` - DISPLAY_DATA_TYPE
     * ``depots`` - DEPOT_SECTION
     * ``demands`` - DEMAND_SECTION
     * ``node_coords`` - NODE_COORD_SECTION
     * ``edge_weights`` - EDGE_WEIGHT_SECTION
     * ``display_data`` - DISPLAY_DATA_SECTION
     * ``edge_data`` - EDGE_DATA_SECTION
     * ``fixed_edges`` - FIXED_EDGES_SECTION
     * ``tours`` - TOUR_SECTION

    For SPECIAL FUNCTION problems, the special function must accept a start
    and an end node and return the weight, distance, or cost of the edge that
    joins them. It can be provided at construction time or simply set on an
    existing object using the ``special`` attribute.

    :param callable special: special function for distance
    :param data: name-value data
    """

    name = F.StringField('NAME')
    comment = F.StringField('COMMENT')
    type = F.StringField('TYPE')
    dimension = F.IntegerField('DIMENSION')

    capacity = F.IntegerField('CAPACITY')
    node_coord_type = F.StringField('NODE_COORD_TYPE')
    edge_weight_type = F.StringField('EDGE_WEIGHT_TYPE')
    display_data_type = F.StringField('DISPLAY_DATA_TYPE')
    edge_weight_format = F.StringField('EDGE_WEIGHT_FORMAT')
    edge_data_format = F.StringField('EDGE_DATA_FORMAT')

    node_coords = F.IndexedCoordinatesField('NODE_COORD_SECTION', dimensions=(2, 3))  # noqa: E501
    edge_data = F.EdgeDataField('EDGE_DATA_SECTION')
    edge_weights = F.MatrixField('EDGE_WEIGHT_SECTION')
    display_data = F.IndexedCoordinatesField('DISPLAY_DATA_SECTION', dimensions=2)  # noqa: E501
    fixed_edges = F.EdgeListField('FIXED_EDGES_SECTION')
    depots = F.DepotsField('DEPOT_SECTION')
    demands = F.DemandsField('DEMAND_SECTION')

    tours = F.ToursField('TOUR_SECTION')

    def __init__(self, special=None, **data):
        super().__init__(**data)
        self._wfunc = None
        self.special = special

    @property
    def special(self):
        """Special distance function.

        Special/custom distance functions must accept two coordinates of
        appropriate dimension and return the distance between them.
        """
        return self._special

    @special.setter
    def special(self, func):
        self._special = func
        self._wfunc = self._create_wfunc(special=func)

    def get_weight(self, start, end):
        """Return the weight of the edge between start and end.

        This method provides a single way to obtain edge weights regardless of
        whether the problem uses an explicit matrix or a distance function.

        :param int start: starting node index
        :param int end: ending node index
        :return: weight of the edge between start and end
        :rtype: float
        """
        return self._wfunc(start, end)

    def is_explicit(self):
        """Check whether the problem specifies edge weights explicitly.

        :return: True if the problem specifies edge weights explicitly
        :rtype: bool
        """
        return self.edge_weight_type == 'EXPLICIT'

    def is_full_matrix(self):
        """Check whether the problem is specified as a full matrix.

        :return: True if the problem is specified as a full matrix
        :rtype: bool
        """
        return self.edge_weight_format == 'FULL_MATRIX'

    def is_weighted(self):
        """Check whether the problem has weighted edges.

        A problem is considered unweighted if neither the EDGE_WEIGHT_FORMAT
        nor the EDGE_WEIGHT_TYPE are defined.

        :return: True if the problem is weighted
        :rtype: bool
        """
        return bool(self.edge_weight_format) or bool(self.edge_weight_type)

    def is_special(self):
        """Check whether the problem is special.

        SPECIAL problems require a special distance function.

        :return: True if the problem requires a special distance function
        :rtype: bool
        """
        return self.edge_weight_type == 'SPECIAL'

    def is_complete(self):
        """Check whether the problem specifies a complete graph.

        :return: True if the problem specifies a complete graph
        :rtype: bool
        """
        return not bool(self.edge_data_format)

    def is_symmetric(self):
        """Check whether the problem is symmetrical.

        .. warning::

            Although a result of ``True`` guarantees symmetry, a value of
            ``False`` merely indicates the *possibliity* for asymmetry. Avoid
            using ``not problem.is_symmetric()`` when possible.

        :return: True if the problem is symmetrical
        :rtype: bool
        """
        return not self.is_full_matrix() and not self.is_special()

    def is_depictable(self):
        """Check whether the problem can be depicted.

        A problem is depictable if it has display data or has node coordinates
        and does not specify NO_DISPLAY.

        :return: True if the problem can be depicted
        :rtype: bool
        """
        if bool(self.display_data):
            return True

        if self.display_data_type == 'NO_DISPLAY':
            return False

        return bool(self.node_coords)

    def trace_tours(self, tours):
        """Return the weights of the given tours.

        Each tour is a list of node indices. The weights returned are the sum
        of the individual weights of the edges in each tour including the final
        edge back to the starting node.

        The list of weights returned parallels the list of tours given so that
        ``weights[i]`` corresponds to ``tours[i]``::

            weights = p.trace_tours(tours)

        :param list tours: one or more lists of node indices
        :return: one weight for each given tour
        :rtype: list
        """
        solutions = []
        for tour in tours:
            edges = utils.pairwise(tour)
            weight = sum(self.get_weight(i, j) for i, j in edges)
            solutions.append(weight)
        return solutions

    def trace_canonical_tour(self):
        """Return the weight of the canonical tour.

        The "canonical tour" uses the nodes in order. This method is present
        primarily for testing and purposes.

        :return: weight of the canonical tour
        :rtype: float
        """
        nodes = list(self.get_nodes())
        return self.trace_tours([nodes])[0]

    def get_nodes(self):
        """Return an iterator over the nodes.

        This method provides a single way to obtain the nodes of a problem
        regardless of how it is specified. However, if the nodes are not
        specified, the EDGE_DATA_FORMAT is not set, and DIMENSION has no value,
        then nodes are undefined.

        :return: nodes
        :rtype: iter
        :raises ValueError: if the nodes are undefined
        """
        if self.node_coords:
            return iter(sorted(self.node_coords))

        if self.display_data:
            return iter(sorted(self.display_data))

        if self.edge_data_format == 'EDGE_LIST':
            nodes = set()
            for a, b in self.edge_data:
                nodes.update({a, b})
            return iter(sorted(nodes))

        if self.edge_data_format == 'ADJ_LIST':
            nodes = set()
            for a, ends in self.edge_data.items():
                nodes.update({a, *ends})
            return iter(sorted(nodes))

        if self.demands:
            return iter(sorted(self.demands))

        try:
            return iter(range(self.dimension))
        except Exception:
            raise ValueError('undefined nodes')

    def get_edges(self):
        """Return an iterator over the edges.

        This method provides a single way to obtain the edges of a problem
        regardless of how it is specified. If the EDGE_DATA_FORMAT is not set
        and the nodes are undefined, then the edges are also undefined.

        :return: edges
        :rtype: iter
        :raises ValueError: if the nodes and therefore the edges are undefined
        """
        if self.edge_data_format == 'EDGE_LIST':
            yield from self.edge_data
        elif self.edge_data_format == 'ADJ_LIST':
            for i, adj in self.edge_data.items():
                yield from ((i, j) for j in adj)
        else:
            yield from itertools.product(self.get_nodes(), self.get_nodes())

    def get_display(self, i):
        """Return the display data for node at index *i*.

        If the problem is not depictable, None is returned instead.

        :param int i: node index
        :return: display data for node i
        """
        if self.is_depictable():
            try:
                return self.display_data[i]
            except (KeyError, TypeError):
                return self.node_coords[i]
        else:
            # TODO: raise an exception instead
            return None

    def get_graph(self, normalize=False):
        """Return a networkx graph instance representing the problem.

        The metadata of the problem is associated with the graph itself.
        Additional problem information is associated with the nodes and edges.
        For example:

        .. code-block:: python

            >>> G = problem.get_graph()
            >>> G.graph
            {'name': None,
             'comment': '14-Staedte in Burma (Zaw Win)',
             'type': 'TSP',
             'dimension': 14,
             'capacity': None}
            >>> G.nodes[1]
            {'coord': (16.47, 96.1),
             'display': None,
             'demand': None,
             'is_depot': False}
            >>> G.edges[1, 2]
            {'weight': 2, 'is_fixed': False}

        If the graph is asymmetric then a :class:`networkx.DiGraph` is
        returned. Optionally, the nodes can be renamed to be sequential and
        zero-indexed.

        :param bool normalize: rename nodes to be zero-indexed
        :return: graph
        :rtype: :class:`networkx.Graph`
        """
        # directed graphs are fundamentally different
        G = networkx.Graph() if self.is_symmetric() else networkx.DiGraph()

        # add basic graph metadata
        G.graph['name'] = self.name
        G.graph['comment'] = self.comment
        G.graph['type'] = self.type
        G.graph['dimension'] = self.dimension
        G.graph['capacity'] = self.capacity

        # set up a map from original node name to new node name
        nodes = list(self.get_nodes())
        if normalize:
            names = {n: i for i, n in enumerate(nodes)}
        else:
            names = {n: n for n in nodes}

        # add every node with some associated metadata
        for n in nodes:
            is_depot = n in self.depots
            G.add_node(names[n], coord=self.node_coords.get(n),
                       display=self.display_data.get(n),
                       demand=self.demands.get(n),
                       is_depot=is_depot)

        # add every edge with some associated metadata
        for a, b in self.get_edges():
            weight = self.get_weight(a, b)
            is_fixed = (a, b) in self.fixed_edges
            G.add_edge(names[a], names[b], weight=weight, is_fixed=is_fixed)

        # return the graph object
        return G

    def _create_wfunc(self, special=None):
        # explicit problems ignore the special function
        if self.is_explicit():
            matrix = self._create_explicit_matrix()
            return lambda i, j: matrix[i, j]

        if self.is_special():
            # use the special weight function
            if special is None:
                raise Exception('missing needed special weight function')
            wfunc = special
        elif self.is_weighted():
            # use a predefined weight function
            wfunc = distances.TYPES[self.edge_weight_type]
        else:
            # unweighted problems
            return lambda i, j: 1

        # Wrap whatever distance function we have so that it takes node
        # indexes instead of directly taking coordinates.
        def adapter(i, j):
            return wfunc(self.node_coords[i], self.node_coords[j])

        return adapter

    def _create_explicit_matrix(self):
        # instantiate the right matrix class for the problem
        m = min(self.get_nodes())
        Matrix = matrix.TYPES[self.edge_weight_format]
        weights = list(itertools.chain(*self.edge_weights))
        return Matrix(weights, self.dimension, min_index=m)
