import os
import bz2
import lzma
import gzip
import codecs
import pickle
import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Tuple
from collections import OrderedDict
from ml4co_kit.utils.graph_utils import GraphData


class FileObject(object):
    def __init__(self, name, mode="r", compression=None):
        self.fp = None
        self.ctype = None
        self.fp_extra = None
        self.open(name, mode=mode, compression=compression)

    def open(self, name, mode="r", compression=None):
        if compression == "use_ext":
            self.get_compression_type(name)
        else:
            self.ctype = compression

        if not self.ctype:
            self.fp = open(name, mode)
        elif self.ctype == "gzip":
            self.fp = gzip.open(name, mode + "t")
        elif self.ctype == "bzip2":
            try:
                self.fp = bz2.open(name, mode + "t")
            except:
                self.fp_extra = bz2.BZ2File(name, mode)
                if mode == "r":
                    self.fp = codecs.getreader("ascii")(self.fp_extra)
                else:
                    self.fp = codecs.getwriter("ascii")(self.fp_extra)
        else:
            self.fp = lzma.open(name, mode=mode + "t")

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None

        if self.fp_extra:
            self.fp_extra.close()
            self.fp_extra = None

        self.ctype = None

    def get_compression_type(self, file_name):
        ext = os.path.splitext(file_name)[1]
        if ext == ".gz":
            self.ctype = "gzip"
        elif ext == ".bz2":
            self.ctype = "bzip2"
        elif ext in (".xz", ".lzma"):
            self.ctype = "lzma"
        else:
            self.ctype = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class MClGraphData(GraphData):
    def __init__(self):
        super(MClGraphData, self).__init__()
        self.nodes_label: np.ndarray = None
        self.ref_nodes_label: np.ndarray = None
        self.sel_nodes_num: np.ndarray = None
        self.ref_sel_nodes_num: np.ndarray = None
        self.self_loop = None
        
    def check_edge_index(self):
        if self.edge_index is not None:
            shape = self.edge_index.shape
            if len(shape) != 2 or shape[0] != 2:
                raise ValueError("The shape of ``edge_index`` must be like (2, E)")

    def check_nodes_label(self, ref: bool):
        nodes_label = self.ref_nodes_label if ref else self.nodes_label
        name = "ref_nodes_label" if ref else "nodes_label"
        if nodes_label is not None:
            if nodes_label.ndim != 1:
                raise ValueError(f"The dimensions of ``{name}`` must be 1.")
        
            if self.nodes_num is not None:
                if len(self.nodes_label) != self.nodes_num:
                    message = (
                        f"The number of nodes in the {name} does not match that of "
                        "the problem. Please check the solution."
                    )
                    raise ValueError(message)
            else:
                self.nodes_num = len(nodes_label)
                  
    def from_adj_martix(self, adj_matrix: np.ndarray, self_loop: bool = True):
        self.self_loop = self_loop
        return super().from_adj_martix(
            adj_matrix=adj_matrix,
            zero_or_one="one",
            type="zero-one",
            self_loop=self.self_loop
        )

    def from_gpickle(
        self, file_path: str, self_loop: bool = True
    ):
        # check file format
        if not file_path.endswith(".gpickle"):
            raise ValueError("Invalid file format. Expected a ``.gpickle`` file.")
        
        # read graph data from .gpickle
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
        graph: nx.Graph

        # nodes num
        self.nodes_num = graph.number_of_nodes()
        
        # edges
        edges = np.array(graph.edges, dtype=np.int64)
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        self.self_loop = self_loop
        if self.self_loop:
            self_loop: np.ndarray = np.arange(self.nodes_num)
            self_loop = self_loop.reshape(-1, 1).repeat(2, axis=1)
            edges = np.concatenate([self_loop, edges], axis=0)
        edges = edges.T

        # use ``from_data``
        self.from_data(edge_index=edges)  
        
    def from_result(self, file_path: str, ref: bool = False):
        # check file format
        if not file_path.endswith(".result"):
            raise ValueError("Invalid file format. Expected a ``.result`` file.")
        
        # read solution from file
        with open(file_path, "r") as f:
            nodes_label = [int(_) for _ in f.read().splitlines()]
        nodes_label = np.array(nodes_label, dtype=np.int64)
        
        # use ``from_data``
        self.from_data(nodes_label=nodes_label, ref=ref)  
    
    def from_data(
        self, 
        edge_index: np.ndarray = None, 
        nodes_label: np.ndarray = None,
        ref: bool = False
    ):
        if edge_index is not None:
            self.edge_index = edge_index
            self.check_edge_index()
        if nodes_label is not None:
            if ref:
                self.ref_nodes_label = nodes_label
            else:
                self.nodes_label = nodes_label
            self.check_nodes_label(ref=ref)
        
    def evaluate(self, calculate_gap: bool = False):
        # solved solution
        if self.sel_nodes_num is None:
            if self.nodes_label is None:
                raise ValueError(
                    "``sel_nodes_num`` cannot be None! You can use solvers based on "
                    "``MClSolver``like ``KaMIS`` to get the ``sel_nodes_num``."
                )
            self.sel_nodes_num = np.sum(self.nodes_label)
    
        # ground truth
        if calculate_gap:
            if self.ref_sel_nodes_num is None:
                if self.ref_nodes_label is None:
                    raise ValueError(
                        "``ref_sel_nodes_num`` cannot be None! You can use solvers based on "
                        "``MClSolver``like ``KaMIS`` to get the ``ref_sel_nodes_num``."
                    )
                self.ref_sel_nodes_num = np.sum(self.ref_nodes_label)
            gap = - (self.sel_nodes_num - self.ref_sel_nodes_num) / self.ref_sel_nodes_num
            return (self.sel_nodes_num, self.ref_sel_nodes_num, gap)
        else:
            return self.sel_nodes_num