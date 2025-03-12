import numpy as np
from ml4co_kit.utils.graph.base import GraphData, np_sparse_to_dense


class MCutGraphData(GraphData):
    def __init__(self):
        super(MCutGraphData, self).__init__()
        self.nodes_label: np.ndarray = None
        self.ref_nodes_label: np.ndarray = None
        self.cut_edge_num: np.ndarray = None
        self.ref_cut_edge_num: np.ndarray = None
        self.self_loop = None
        
    def _check_edge_index(self):
        if self.edge_index is not None:
            shape = self.edge_index.shape
            if len(shape) != 2 or shape[0] != 2:
                raise ValueError("The shape of ``edge_index`` must be like (2, E)")

    def _check_nodes_label(self, ref: bool):
        nodes_label = self.ref_nodes_label if ref else self.nodes_label
        name = "ref_nodes_label" if ref else "nodes_label"
        if nodes_label is not None:
            if nodes_label.ndim != 1:
                raise ValueError(f"The dimensions of ``{name}`` must be 1.")
        
            if self.nodes_num is not None:
                if len(nodes_label) != self.nodes_num:
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

    def from_data(
        self, 
        edge_index: np.ndarray = None, 
        nodes_label: np.ndarray = None,
        ref: bool = False
    ):
        if edge_index is not None:
            self.edge_index = edge_index
            self._check_edge_index()
        if nodes_label is not None:
            if ref:
                self.ref_nodes_label = nodes_label
            else:
                self.nodes_label = nodes_label
            self._check_nodes_label(ref=ref)
        
    def evaluate(self, calculate_gap: bool = False):
        # solved solution
        if self.cut_edge_num is None:
            if self.nodes_label is None:
                raise ValueError(
                    "``cut_edge_num`` cannot be None! You can use solvers based on "
                    "``MCutSolver``like ``KaMIS`` to get the ``cut_edge_num``."
                )
            adj_matrix = np_sparse_to_dense(nodes_num=self.nodes_num, edge_index=self.edge_index)
            self.cut_edge_num = 0
            set_A, set_B = [], []
            for node, label in enumerate(self.nodes_label):
                if label == 1:
                    set_A.append(node)
                else:
                    set_B.append(node)
            for i in set_A:
                for j in set_B:
                    if adj_matrix[i, j] == 1:
                        self.cut_edge_num += 1
    
        # ground truth
        if calculate_gap:
            if self.ref_cut_edge_num is None:
                if self.ref_nodes_label is None:
                    raise ValueError(
                        "``ref_cut_edge_num`` cannot be None! You can use solvers based on "
                        "``MCutSolver``like ``KaMIS`` to get the ``ref_cut_edge_num``."
                    )
                adj_matrix = np_sparse_to_dense(nodes_num=self.nodes_num, edge_index=self.edge_index)
                self.ref_cut_edge_num = 0
                set_A, set_B = [], []
                for node, label in enumerate(self.ref_nodes_label):
                    if label == 1:
                        set_A.append(node)
                    else:
                        set_B.append(node)
                for i in set_A:
                    for j in set_B:
                        if adj_matrix[i, j] == 1:
                            self.ref_cut_edge_num += 1
            gap = - (self.cut_edge_num - self.ref_cut_edge_num) / self.ref_cut_edge_num * 100
            return (self.cut_edge_num, self.ref_cut_edge_num, gap)
        else:
            return self.cut_edge_num