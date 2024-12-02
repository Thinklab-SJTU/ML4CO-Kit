from . import models


def load(filepath, problem_class=None, special=None):
    """Load a problem at the given filepath.

    :param str filepath: path to a TSPLIB problem file
    :param type problem_class: special/custom problem class
    :param callable special: special/custom distance function
    :return: problem instance
    :rtype: :class:`~Problem`
    """
    with open(filepath) as f:
        return read(f, special=special, problem_class=problem_class)


def read(f, problem_class=None, special=None):
    """Read a problem from a file-like object.

    :param file f: file-like object
    :param type problem_class: special/custom problem class
    :param callable special: special/custom distance function
    :return: problem instance
    :rtype: :class:`~Problem`
    """
    return parse(f.read(), special=special, problem_class=problem_class)


def parse(text, problem_class=None, special=None):
    """Load a problem from raw text.

    :param str text: text of a TSPLIB problem
    :param type problem_class: special/custom problem class
    :param callable special: special/custom distance function
    :return: problem instance
    :rtype: :class:`~Problem`
    """
    Problem = problem_class or models.StandardProblem
    return Problem.parse(text, special=special)


def load_problem(filepath, special=None):
    """Load a problem at the given filepath.

    :param str filepath: path to a TSPLIB problem file
    :param callable special: special/custom distance function
    :return: problem instance
    :rtype: :class:`~Problem`
    """
    return load(filepath, special=special)


def load_solution(filepath):
    """Load a solution at the given filepath.

    :param str filepath: path to a TSPLIB solution file
    :return: solution instance
    :rtype: :class:`~Solution`
    """
    return load(filepath)


def load_unknown(filepath):
    """Load any TSPLIB file.

    This is particularly useful when you do not know in advance
    whether the file contains a problem or a solution.

    :param str filepath: path to a TSPLIB problem file
    :return: either a problem or solution instance
    """
    return load(filepath)


def load_problem_fromstring(text, special=None):
    """Load a problem from raw text.

    :param str text: text of a TSPLIB problem
    :param callable special: special/custom distance function
    :return: problem instance
    :rtype: :class:`~Problem`
    """
    return parse(text, special=special)


def load_solution_fromstring(text):
    """Load a solution from raw text.

    :param str text: text of a TSPLIB solution
    :return: solution instance
    :rtype: :class:`~Solution`
    """
    return parse(text)


def load_unknown_fromstring(text):
    """Load any problem/solution from raw text.

    This is particularly useful when you do not know in advance
    whether the file contains a problem or a solution.

    :param str text: text of a TSPLIB problem/solution
    :return: either a problem or solution instance
    """
    return parse(text)
