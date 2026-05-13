r"""
EDA Placement.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.


import numpy as np
from typing import Union, List, Tuple
from ml4co_kit.task.base import TASK_TYPE, TaskBase
from ml4co_kit.task.eda.c_edap_helper import EDAHelper


class EDAPTask(TaskBase):
    def __init__(
        self,
        precision: Union[np.float32, np.float64] = np.float32,
        grid_bins: tuple = (224, 224),
        wirelength_weight: float = 1.0,
        congestion_weight: float = 1.0,
    ):
        # Super Initialization
        super(EDAPTask, self).__init__(
            task_type=TASK_TYPE.EDAP,
            minimize=True,
            precision=precision
        )

        # Initialize Attributes (Weight)
        self.bin_cols, self.bin_rows = grid_bins
        self.wirelength_weight = wirelength_weight
        self.congestion_weight = congestion_weight

        # Initialize Attributes (Basic)
        self.die: np.ndarray = None # [R, (xl, yl, xh, yh)]
        self.cells: np.ndarray = None # [N, (width, height)]
        self.cells_num: int = None
        self.macro_mask: np.ndarray = None # [N,]
        self.helper = EDAHelper()

        # List of nets, each net is a 2D array of shape
        # [Np, (macro_idx, offset_x, offset_y)]
        self.nets: List[np.ndarray] = None

    def _check_die_dim(self):
        if self.die.ndim != 2 or self.die.shape[1] != 4:
            raise ValueError("``die`` should be a 2D array of shape (block number, 4).")

    def _check_cells_dim(self):
        if self.cells.ndim != 2 or self.cells.shape[1] != 2:
            raise ValueError("``cells`` should be a 2D array of shape (N, 2).")

    def _check_macro_mask_dim(self):
        if self.macro_mask.ndim != 1:
            raise ValueError("Macro mask should be a 1D array.")

    def _check_nets_dim(self):
        if self.nets is not None:
            for net in self.nets:
                if net.ndim != 2 or net.shape[1] != 3:
                    raise ValueError("Each net should be a 2D array of shape (Np, 3).")

    def _check_sol_dim(self):
        if self.sol.ndim != 2:
            raise ValueError("Solution should be a 2D array.")
    
    def _check_ref_sol_dim(self):
        if self.ref_sol.ndim != 2:
            raise ValueError("Reference solution should be a 2D array.")

    def from_data(
        self,
        die: np.ndarray = None,
        cells: np.ndarray = None,
        cells_num: int = None,
        macro_mask: np.ndarray = None,
        nets: List[np.ndarray] = None,
        sol: np.ndarray = None,
        ref: bool = False,
        name: str = None,
    ):
        # Set Attributes and Check Dimensions
        if cells_num is not None:
            self.cells_num = cells_num
        if die is not None:
            self.die = die.astype(self.precision)
            self._check_die_dim()
        if cells is not None:
            self.cells = cells.astype(self.precision)
            self._check_cells_dim()
            self.cells_num = self.cells.shape[0]
        if macro_mask is not None:
            self.macro_mask = macro_mask.astype(np.bool_)
            self._check_macro_mask_dim()
        if nets is not None:
            self.nets = nets
            self._check_nets_dim()
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()

        # Set Name if Provided
        if name is not None:
            self.name = name

    def check_constraints(self, sol: np.ndarray) -> bool:
        # Check if die and cells are None
        if self.die is None or self.cells is None:
            return False
        
        # Boundary uses the row/subrow rectangles; overlap is checked geometrically.
        inside_die, overlap = self.helper.check_constraints(
            sol, self.die, self.cells, self.macro_mask
        )
        if inside_die == False:
            print("Warning: Some cells are outside the die!")
        return True if overlap == 0 else False

    def evaluate(
        self, sol: np.ndarray, check_constr: bool = True
    ) -> Tuple[np.floating, np.ndarray, np.ndarray]:
        # Check Constraints
        if check_constr and not self.check_constraints(sol):
            raise ValueError("Invalid solution!")

        # Evaluate
        wirelength, congestion = self.helper.evaluate(
            sol, self.nets, self.die, self.bin_cols, self.bin_rows
        )
        congestion: np.ndarray
        max_congestion = np.max(congestion).item()
        avg_congestion = np.mean(congestion).item()
        return wirelength, max_congestion, avg_congestion

    def evaluate_w_gap(self, check_constr: bool = True):
        info = (
            "Due to the particularity of EDA problems, "
            "which involve multiple evaluation metrics, "
            "we do not recommend using gap as an evaluation method."
        )
        raise NotImplementedError(info)
   