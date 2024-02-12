import os
import pickle
import pathlib
import numpy as np
import networkx as nx


class MISSolver:
    def __init__(self) -> None:
        self.solver_type = None
        self.weighted = None
        self.time_limit = 60.0
        self.nodes_num = None
        self.node_labels = None
        self.gt_node_labels = None
        self.sel_nodes_num = None
        self.gt_sel_nodes_num = None
        self.edges = None
        
    def from_folder(self, folder: str, weighted: bool=False):
        self.folder = folder
        self.solve_folder = os.path.join(self.folder, "solve")
        read_label = True if os.path.exists(self.solve_folder) else False
        files = os.listdir(folder)
        self.nodes_num = list()
        self.gt_node_labels = list()
        self.gt_sel_nodes_num = list()
        self.edges = list()
        for filename in files:
            if not filename.endswith(".gpickle"):
                continue
            file_path = os.path.join(self.folder, filename)
            with open(file_path, "rb") as f:
                graph = pickle.load(f)
            graph: nx.Graph
            # nodes num
            nodes_num = graph.number_of_nodes()
            # node lables
            if not read_label:
                node_labels = [_[1] for _ in graph.nodes(data='label')]
                if node_labels is not None and node_labels[0] is not None:
                    node_labels = np.array(node_labels, dtype=np.int64)
                else:
                    node_labels = np.zeros(nodes_num, dtype=np.int64)
                    edges = np.array(graph.edges, dtype=np.int64)
            else:
                solve_filename = filename.replace('.gpickle', \
                    f"_{'weighted' if weighted else 'unweighted'}.result")
                solve_file_path = os.path.join(self.solve_folder, solve_filename)
                with open(solve_file_path, 'r') as f:
                    node_labels = [int(_) for _ in f.read().splitlines()]
                node_labels = np.array(node_labels, dtype=np.int64)
                if node_labels.shape[0] != nodes_num:
                    message = "The number of nodes in the solution result does not match the number " 
                    message += "of nodes in the problem. Please check the solution result."
                    raise ValueError(message)
            # edges
            edges = np.array(graph.edges, dtype=np.int64)
            edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
            self_loop = np.arange(nodes_num).reshape(-1, 1).repeat(2, axis=1)
            edges = np.concatenate([edges, self_loop], axis=0)
            edges = edges.T
            # add to list
            self.nodes_num.append(nodes_num)
            self.gt_node_labels.append(node_labels)
            self.gt_sel_nodes_num.append(np.count_nonzero(node_labels))
            self.edges.append(edges)
         
    @staticmethod
    def __prepare_graph(g: nx.Graph, weighted = False):
        raise NotImplementedError("__prepare_graph is required to implemented in subclass")
    
    def prepare_instances(self, instance_directory: pathlib.Path, 
                           cache_directory: pathlib.Path):
        raise NotImplementedError("prepare_instances is required to implemented in subclass")
    
    def solve(self, solve_data_path: pathlib.Path, results_path: pathlib.Path):
        raise NotImplementedError("solve is required to implemented in subclass")