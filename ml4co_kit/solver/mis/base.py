import os
import pickle
import pathlib
import numpy as np
import networkx as nx
from tqdm import tqdm
from ml4co_kit.utils.mis_utils import sat_to_mis_graph
from collections import OrderedDict


class MISSolver:
    def __init__(self, solver_type: str = None) -> None:
        self.solver_type = solver_type
        self.weighted = None
        self.time_limit = 60.0
        self.nodes_num = None
        self.node_labels = None
        self.ref_node_labels = None
        self.sel_nodes_num = None
        self.ref_sel_nodes_num = None
        self.edges = None

    def check_sel_nodes_num_not_none(self):
        if self.sel_nodes_num is None:
            message = (
                "``sel_nodes_num`` cannot be None, please use the method "
                "``solve`` to obtain solutions and use the method ``from_gpickle_folder`` to "
                "get ``sel_nodes_num``."
            )
            raise ValueError(message)

    def check_ref_sel_nodes_num_not_none(self):
        if self.ref_sel_nodes_num is None:
            raise ValueError(
                "``ref_sel_nodes_num`` cannot be None, please use KaMIS to obtain it."
            )

    def from_gpickle_folder(
        self,
        folder: str,
        solve_folder: str = None,
        weighted: bool = False,
        ref: bool = False,
    ):
        if solve_folder is None:
            solve_folder = os.path.join(folder, "solve")
        read_label = True if os.path.exists(solve_folder) else False
        files = os.listdir(folder)
        self.nodes_num = list()
        record_node_labels = OrderedDict()
        record_sel_nodes_num = OrderedDict()
        self.edges = list()
        for filename in files:
            # check the file format
            if not filename.endswith(".gpickle"):
                continue

            # read graph data from .gpickle
            file_path = os.path.join(folder, filename)
            with open(file_path, "rb") as f:
                graph = pickle.load(f)
            graph: nx.Graph

            # nodes num
            nodes_num = graph.number_of_nodes()

            # node labels
            if not read_label:
                node_labels = [_[1] for _ in graph.nodes(data="label")]
                if node_labels is not None and node_labels[0] is not None:
                    node_labels = np.array(node_labels, dtype=np.int64)
                else:
                    node_labels = np.zeros(nodes_num, dtype=np.int64)
                    edges = np.array(graph.edges, dtype=np.int64)
            else:
                solve_filename = filename.replace(
                    ".gpickle", f"_{'weighted' if weighted else 'unweighted'}.result"
                )
                solve_file_path = os.path.join(solve_folder, solve_filename)
                with open(solve_file_path, "r") as f:
                    node_labels = [int(_) for _ in f.read().splitlines()]
                node_labels = np.array(node_labels, dtype=np.int64)
                if node_labels.shape[0] != nodes_num:
                    message = (
                        "The number of nodes in the solution does not match that of"
                        "the problem. Please check the solution."
                    )
                    raise ValueError(message)

            # edges
            edges = np.array(graph.edges, dtype=np.int64)
            edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
            self_loop = np.arange(nodes_num).reshape(-1, 1).repeat(2, axis=1)
            edges = np.concatenate([edges, self_loop], axis=0)
            edges = edges.T

            # add to the list/dict
            self.nodes_num.append(nodes_num)
            record_node_labels[filename] = node_labels
            record_sel_nodes_num[filename] = np.count_nonzero(node_labels)
            self.edges.append(edges)

        if ref:
            self.ref_node_labels = record_node_labels
            self.ref_sel_nodes_num = record_sel_nodes_num
        else:
            self.node_labels = record_node_labels
            self.sel_nodes_num = record_sel_nodes_num

    def read_ref_sel_nodes_num_from_txt(self, file_path: str):
        self.ref_sel_nodes_num = OrderedDict()

        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                split_line = line.split(": ")
                filename = split_line[0]
                ref_sel_nodes_num = int(split_line[1])
                self.ref_sel_nodes_num[filename] = ref_sel_nodes_num

    @staticmethod
    def __prepare_graph(g: nx.Graph, weighted=False):
        raise NotImplementedError(
            "``__prepare_graph`` is required to implemented in subclasses."
        )

    def prepare_instances(
        self, instance_directory: pathlib.Path, cache_directory: pathlib.Path
    ):
        raise NotImplementedError(
            "``prepare_instances`` is required to implemented in subclasses."
        )

    def solve(self, src: pathlib.Path, out: pathlib.Path = None):
        raise NotImplementedError(
            "The method ``solve`` is required to implemented in subclasses."
        )

    def evaluate(self, calculate_gap: bool = False):
        self.check_sel_nodes_num_not_none()
        sel_nodes_num = np.array(list(self.sel_nodes_num.values()))

        if not calculate_gap:
            return_dict = {
                "avg_sel_nodes_num": np.average(sel_nodes_num),
            }
            return return_dict

        # ref_sel_nodes_num
        self.check_ref_sel_nodes_num_not_none()
        ref_sel_nodes_num = np.array(list(self.ref_sel_nodes_num.values()))

        # calculate gap
        gaps = list()
        for filename, ref in self.ref_sel_nodes_num.items():
            solved = self.sel_nodes_num[filename]
            gap = (solved - ref) / ref * 100
            gaps.append(gap)
        gaps = np.array(gaps)

        return_dict = {
            "avg_sel_nodes_num": np.average(sel_nodes_num),
            "avg_ref_sel_nodes_num": np.average(ref_sel_nodes_num),
            "avg_gap": np.average(gaps),
        }
        return return_dict
