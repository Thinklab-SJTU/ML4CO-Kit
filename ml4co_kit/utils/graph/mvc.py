import numpy as np
from ml4co_kit.utils.graph.base import GraphData


class MVCGraphData(GraphData):
    def __init__(self):
        super(MVCGraphData, self).__init__()
        self.nodes_label: np.ndarray = None
        self.ref_nodes_label: np.ndarray = None
        self.sel_nodes_num: np.ndarray = None
        self.ref_sel_nodes_num: np.ndarray = None
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
        
    def evaluate(self, calculate_gap: bool = False, check_constraint: bool = False):
        # check constraint
        if check_constraint:
            self.check_constraint(ref=False)
            if calculate_gap:
                self.check_constraint(ref=True)
            
        # solved solution
        if self.sel_nodes_num is None:
            if self.nodes_label is None:
                raise ValueError(
                    "``sel_nodes_num`` cannot be None! You can use solvers based on "
                    "``MVCSolver``like ``KaMIS`` to get the ``sel_nodes_num``."
                )
            self.sel_nodes_num = np.sum(self.nodes_label)
    
        # ground truth
        if calculate_gap:
            if self.ref_sel_nodes_num is None:
                if self.ref_nodes_label is None:
                    raise ValueError(
                        "``ref_sel_nodes_num`` cannot be None! You can use solvers based on "
                        "``MVCSolver``like ``KaMIS`` to get the ``ref_sel_nodes_num``."
                    )
                self.ref_sel_nodes_num = np.sum(self.ref_nodes_label)
            gap = (self.sel_nodes_num - self.ref_sel_nodes_num) / self.ref_sel_nodes_num * 100
            return (self.sel_nodes_num, self.ref_sel_nodes_num, gap)
        else:
            return self.sel_nodes_num
        
    def check_constraint(self, ref: bool):
        self._check_nodes_label(ref=ref)
        sol = self.ref_nodes_label if ref else self.nodes_label
        if sol is not None:
            src_dst = sol[self.edge_index[0]] + sol[self.edge_index[1]]
            if src_dst.min() == 0:
                raise ValueError(
                    "The solution does not conform to constraint!"
                )