import pathlib
import networkx as nx


class MISSolver:
    def __init__(self) -> None:
        self.solver_type = None
        self.weighted = None
        self.time_limit = 60.0
    
    @staticmethod
    def __prepare_graph(g: nx.Graph, weighted = False):
        raise NotImplementedError("__prepare_graph is required to implemented in subclass")
    
    def prepare_instances(self, instance_directory: pathlib.Path, 
                           cache_directory: pathlib.Path):
        raise NotImplementedError("prepare_instances is required to implemented in subclass")
    
    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path):
        raise NotImplementedError("solve is required to implemented in subclass")