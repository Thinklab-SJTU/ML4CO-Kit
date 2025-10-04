import json
import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm


class ConstraintLanguage:
    """
    Class to represent a fixed Constraint Language.
    """

    def __init__(self, domain_size: int, relations: dict):
        """
        Initializes a Constraint Language.

        :param domain_size: Size of the underlying domain.
        :param relations: A dictionary specifying the relations of the language.
                         Each key is a relation name, and each value is a list of tuples representing the relation.
                         Example: {'XOR': [[0, 1], [1, 0]], 'AND': [[1, 1]]}
        """
        self.domain_size = domain_size
        self.relations = relations
        self.relation_names = list(relations.keys())

        # Compute characteristic matrices for each relation
        self.relation_matrices = {}
        for name, relation in self.relations.items():
            matrix = np.zeros((self.domain_size, self.domain_size), dtype=np.float32)
            for pair in relation:
                matrix[pair[0], pair[1]] = 1.0
            self.relation_matrices[name] = matrix

    def save(self, path: str):
        """
        Saves the constraint language to a JSON file.

        :param path: Path to save the JSON file.
        """
        with open(path, 'w') as f:
            json.dump({'domain_size': self.domain_size, 'relations': self.relations}, f, indent=4)

    @staticmethod
    def load(path: str):
        """
        Loads a constraint language from a JSON file.

        :param path: Path to the JSON file.
        :return: A ConstraintLanguage object.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return ConstraintLanguage(data['domain_size'], data['relations'])

    @staticmethod
    def get_coloring_language(domain_size: int):
        """
        Creates a constraint language for graph coloring.

        :param domain_size: Number of colors.
        :return: A ConstraintLanguage object for graph coloring.
        """
        def get_neq_relation(d: int):
            """
            Generates the 'not equal' relation for graph coloring.

            :param d: Number of colors.
            :return: A list of tuples representing the 'not equal' relation.
            """
            return [[i, j] for i in range(d) for j in range(d) if i != j]

        return ConstraintLanguage(
            domain_size=domain_size,
            relations={'NEQ': get_neq_relation(domain_size)}
        )


# Define constant constraint languages for common problems
coloring_language = ConstraintLanguage(
    domain_size=3,
    relations={'NEQ': [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]}
)

independent_set_language = ConstraintLanguage(
    domain_size=2,
    relations={'NAND': [[0, 0], [0, 1], [1, 0]]}
)

max_2sat_language = ConstraintLanguage(
    domain_size=2,
    relations={
        'OR': [[0, 1], [1, 0], [1, 1]],
        'IMPL': [[0, 0], [0, 1], [1, 1]],
        'NAND': [[0, 0], [0, 1], [1, 0]]
    }
)

max_cut_weighted_language = ConstraintLanguage(
    domain_size=2,
    relations={'EQ': [[1, 1], [0, 0]], 'NEQ': [[1, 0], [0, 1]]}
)


class CSPInstance:
    """
    Class to represent a CSP instance.
    """

    def __init__(
        self, language: ConstraintLanguage, n_variables: int, 
        clauses: dict, clause_weights: dict = None, name: str = None
    ):
        """
        Initializes a CSP instance.

        :param language: A ConstraintLanguage object.
        :param n_variables: Number of variables.
        :param clauses: A dictionary specifying the clauses for each relation in the language.
                        Example: {'XOR': [[1, 2], [5, 4], [3, 1]], 'AND': [[1, 4], [2, 5]]}
        :param clause_weights: Optional dictionary specifying weights for each clause.
        :param name: Optional name for the instance.
        """
        self.language = language
        self.n_variables = n_variables
        self.clauses = {relation: np.int32(clause_list) for relation, clause_list in clauses.items()}
        self.name = name

        if clause_weights is not None:
            self.weighted = True
            self.clause_weights = {relation: np.float32(weights) for relation, weights in clause_weights.items()}
        else:
            self.weighted = False

        # Compute number of clauses and degree of each variable
        all_clauses = list(itertools.chain.from_iterable(clauses.values()))
        variables, counts = np.unique(all_clauses, return_counts=True)
        degrees = np.zeros(shape=(n_variables), dtype=np.int32)
        for var, count in zip(variables, counts):
            degrees[var] = count

        self.degrees = degrees
        self.n_clauses = len(all_clauses)

    def count_conflicts(self, assignment: list):
        """
        Counts the number of unsatisfied clauses in the instance given a variable assignment.

        :param assignment: A list of integers representing the variable assignment.
        :return: Number of unsatisfied clauses.
        """
        conflicts = 0
        for relation, matrix in self.language.relation_matrices.items():
            valid = np.float32([matrix[assignment[u], assignment[v]] for u, v in self.clauses[relation]])
            has_conflict = 1.0 - valid
            if self.weighted:
                has_conflict *= self.clause_weights[relation]
            conflicts += np.sum(has_conflict)
        return int(conflicts)

    @staticmethod
    def merge(instances):
        """
        Merges multiple CSP instances into one.

        :param instances: List of CSP instances to merge.
        :return: A single merged CSP instance.
        """
        language: ConstraintLanguage = instances[0].language
        clauses = {relation: [] for relation in language.relation_names}
        n_variables = 0

        for instance in instances:
            for relation in language.relation_names:
                shifted_clauses = instance.clauses[relation] + n_variables
                clauses[relation].append(shifted_clauses)
            n_variables += instance.n_variables

        clauses = {relation: np.vstack(clause_list) for relation, clause_list in clauses.items()}

        if instances[0].weighted:
            weights = {relation: np.hstack([inst.clause_weights[relation] for inst in instances]) for relation in language.relation_names}
        else:
            weights = None

        return CSPInstance(language, n_variables, clauses, weights)

    @staticmethod
    def batch_instances(instances: list, batch_size):
        """
        Splits a list of CSP instances into batches.

        :param instances: List of CSP instances.
        :param batch_size: Number of instances per batch.
        :return: List of batches, where each batch is a merged CSP instance.
        """
        n_instances = len(instances)
        n_batches = int(np.ceil(n_instances / batch_size))
        batches = []

        print('Combining instances into batches...')
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min(start + batch_size, n_instances)
            batch_instance = CSPInstance.merge(instances[start:end])
            batches.append(batch_instance)

        return batches

    @staticmethod
    def generate_random(
        n_variables: int, n_clauses: int, 
        language: ConstraintLanguage, weighted: bool = False
    ):
        """
        Generates a random CSP instance.

        :param n_variables: Number of variables.
        :param n_clauses: Number of clauses.
        :param language: ConstraintLanguage object.
        :param weighted: Whether to assign random weights to clauses.
        :return: A random CSP instance.
        """
        variables = list(range(n_variables))
        clauses = {relation: [] for relation in language.relation_names}
        relations = np.random.choice(language.relation_names, n_clauses)

        for i in range(n_clauses):
            clause = list(np.random.choice(variables, 2, replace=False))
            relation = relations[i]
            clauses[relation].append(clause)

        if weighted:
            clause_weights = {
                relation: np.random.uniform(size=len(clauses[relation])) for \
                    relation in language.relation_names
            }
        else:
            clause_weights = None

        return CSPInstance(language, n_variables, clauses, clause_weights)

    @staticmethod
    def graph_to_csp_instance(
        graph: nx.Graph, language: ConstraintLanguage, 
        relation_name: str, name: str = None
    ):
        """
        :param graph: A NetworkX graphs
        :param language: A Constraint Language
        :param relation_name: The relation name to assign to each edge
        :return: A CSP Instance representing the graph
        """
        adj = nx.linalg.adjacency_matrix(graph)
        n_variables = adj.shape[0]
        clauses = {relation_name: np.int32(graph.edges())}

        instance = CSPInstance(language, n_variables, clauses, name=name)
        return instance

    @staticmethod
    def graph_to_weighted_mc_instance(graph: nx.Graph, name: str = None):
        """
        :param graph: A NetworkX graphs
        :param language: A Constraint Language
        :param relation_name: The relation name to assign to each edge
        :return: A CSP Instance representing the graph
        """
        adj = nx.linalg.adjacency_matrix(graph)
        n_variables = adj.shape[0]
        clauses = {'EQ': [], 'NEQ': []}
        for u, v, w in graph.edges(data='weight'):
            rel = 'NEQ' if w > 0 else 'EQ'
            clauses[rel].append([u, v])

        instance = CSPInstance(max_cut_weighted_language, n_variables, clauses, name=name)
        return instance

    @staticmethod
    def cnf_to_instance(formula, clause_weights=None):
        """
        :param formula: A 2-cnf formula represented as a list of lists of ints.
                        I.e. ((X1 or X2) and (not X2 or X3)) is [[1, 2], [-2, 3]]
        :return: A CSP instance that represents the formula
        """

        def clause_type(clause):
            # returns the relation type for a given clause
            if clause[0] * clause[1] < 0:
                return 'IMPL'
            elif clause[0] > 0:
                return 'OR'
            else:
                return 'NAND'

        def normalize_2SAT_clauses(formula):
            # Transforms clauses of form [v, -u] to [-u, v]. This unifies the direction of all implication clauses.
            fill_monom_clause = lambda c: [c[0], c[0]] if len(c) == 1 else c
            filled_formula = list(map(fill_monom_clause, formula))
            normalize_impl_clause = lambda c: [c[1], c[0]] if clause_type(c) == 'IMPL' and c[0] > 0 else c
            normed_formula = list(map(normalize_impl_clause, filled_formula))
            return normed_formula

        formula = normalize_2SAT_clauses(formula)

        clauses = {t: [] for t in {'OR', 'IMPL', 'NAND'}}

        weighted = clause_weights is not None
        if weighted:
            weights = {t: [] for t in {'OR', 'IMPL', 'NAND'}}
        else:
            weights = None

        for i, c in enumerate(formula):
            u = abs(c[0]) - 1
            v = abs(c[1]) - 1
            t = clause_type(c)
            clauses[t].append([u, v])
            if weighted:
                weights[t].append(clause_weights[i])

        n_variables = np.max([np.max(np.abs(clause)) for clause in formula])

        instance = CSPInstance(max_2sat_language, n_variables, clauses, clause_weights=weights)
        return instance