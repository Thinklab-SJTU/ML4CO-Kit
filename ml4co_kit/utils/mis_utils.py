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


class CNF(object):
    def __init__(
        self,
        from_file = None,
        from_fp = None,
        comment_lead = ["c"],
    ):
        self.nv = 0
        self.clauses = []
        self.comments = []

        if from_file:
            self.from_file(from_file, comment_lead, compressed_with="use_ext")
        elif from_fp:
            self.from_fp(from_fp, comment_lead)

    def __repr__(self):
        """
        State reproducible string representaion of object.
        """
        s = self.to_dimacs().replace("\n", "\\n")
        return f'CNF(from_string="{s}")'

    def from_file(self, fname, comment_lead = ["c"], compressed_with = "use_ext"):
        with FileObject(fname, mode="r", compression=compressed_with) as fobj:
            self.from_fp(fobj.fp, comment_lead)

    def from_fp(self, file_pointer, comment_lead=["c"]):
        self.nv = 0
        self.clauses = []
        self.comments = []
        comment_lead = set(["p"]).union(set(comment_lead))

        for line in file_pointer:
            line = line.rstrip()
            if line:
                if line[0] not in comment_lead:
                    self.clauses.append(list(map(int, line.split()[:-1])))
                elif not line.startswith("p cnf "):
                    self.comments.append(line)

        self.nv = max(
            map(
                lambda cl: max(map(abs, cl)),
                itertools.chain.from_iterable([[[self.nv]], self.clauses]),
            )
        )


def sat_to_mis_graph(sat_path: str) -> Tuple[nx.Graph, int]:
    cnf = CNF(sat_path)
    nv = cnf.nv
    clauses_num = len(cnf.clauses)
    clauses = list(filter(lambda x: x, cnf.clauses))
    ind = {k: [] for k in np.concatenate([np.arange(1, nv + 1), -np.arange(1, nv + 1)])}
    edges = []
    for i, clause in enumerate(clauses):
        a = clause[0]
        b = clause[1]
        c = clause[2]
        aa = 3 * i + 0
        bb = 3 * i + 1
        cc = 3 * i + 2
        ind[a].append(aa)
        ind[b].append(bb)
        ind[c].append(cc)
        edges.append((aa, bb))
        edges.append((aa, cc))
        edges.append((bb, cc))
    for i in np.arange(1, nv + 1):
        for u in ind[i]:
            for v in ind[-i]:
                edges.append((u, v))
    graph = nx.from_edgelist(edges)
    return graph, clauses_num


def cnf_to_gpickle(file_path: str, save_path: str = None) -> Tuple[nx.Graph, int]:
    # check the file format
    if not file_path.endswith(".cnf"):
        raise ValueError("Invalid file format. Expected a ``.cnf`` file.")
    if save_path is not None:
        if not save_path.endswith(".gpickle"):
            raise ValueError("Invalid file format. Expected a ``.gpickle`` file.")
    
    # convert sat to mis graph
    graph, clauses_num = sat_to_mis_graph(sat_path=file_path)
    if save_path is not None:
        if not save_path.endswith(".gpickle"):
            save_path += ".gpickle"
        with open(save_path, "wb") as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    return graph, clauses_num
        

def cnf_folder_to_gpickle_folder(cnf_folder: str, gpickle_foler: str):
    # check the folder
    mis_graph_save_dir = os.path.join(gpickle_foler, "instance")
    if not os.path.exists(mis_graph_save_dir):
        os.makedirs(mis_graph_save_dir)
        
    # cnf to gpickle
    ref_dict = OrderedDict()
    cnf_files = os.listdir(cnf_folder)
    cnf_files.sort()
    for cnf_file in tqdm(cnf_files, desc=f"Processing files in {cnf_folder}"):
        file_path = os.path.join(cnf_folder, cnf_file)
        gpickle_file = cnf_file.replace(".cnf", ".gpickle")
        save_path = os.path.join(mis_graph_save_dir, gpickle_file)
        _, clauses_num = cnf_to_gpickle(file_path, save_path)
        ref_dict[gpickle_file] = clauses_num
    
    # save the ref. solution
    ref_save_path = os.path.join(gpickle_foler, "ref_solution.txt")
    with open(ref_save_path, 'w') as ref_file:
        for gpickle_file, clauses_num in ref_dict.items():
            ref_file.write(f"{gpickle_file}: {clauses_num}\n")
            

class MISGraphData(GraphData):
    def __init__(self):
        super(MISGraphData, self).__init__()
        self.nodes_label: np.ndarray = None
        self.ref_nodes_label: np.ndarray = None
        self.sel_nodes_num: np.ndarray = None
        self.ref_sel_nodes_num: np.ndarray = None
    
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
                self.nodes_num = len(self.nodes_label)
                  
    def from_adj_martix(self, adj_matrix: np.ndarray, self_loop: bool = True):
        return super().from_adj_martix(
            adj_matrix=adj_matrix,
            zero_or_one="one",
            type="zero-one",
            self_loop=self_loop
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
        if self_loop:
            self_loop = np.arange(self.nodes_num).reshape(-1, 1).repeat(2, axis=1)
            edges = np.concatenate([edges, self_loop], axis=0)
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
                    "``MISSolver``like ``KaMIS`` to get the ``sel_nodes_num``."
                )
            self.sel_nodes_num = np.sum(self.nodes_label)
    
        # ground truth
        if calculate_gap:
            if self.ref_sel_nodes_num is None:
                if self.ref_nodes_label is None:
                    raise ValueError(
                        "``ref_sel_nodes_num`` cannot be None! You can use solvers based on "
                        "``MISSolver``like ``KaMIS`` to get the ``ref_sel_nodes_num``."
                    )
                self.ref_sel_nodes_num = np.sum(self.ref_nodes_label)
            gap = - (self.sel_nodes_num - self.ref_sel_nodes_num) / self.ref_sel_nodes_num
            return (self.sel_nodes_num, self.ref_sel_nodes_num, gap)
        else:
            return self.sel_nodes_num