r"""
EDA Placement.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.


import os
import pathlib
import numpy as np
from typing import Union, List, Tuple
from ml4co_kit.task.eda.base import EDA_BENCH
from ml4co_kit.task.base import TASK_TYPE, TaskBase
from ml4co_kit.task.eda.c_edap_helper import EDAHelper
from ml4co_kit.task.eda.c_edap_reader import ISPD2005Reader


class EDAPTask(TaskBase):
    def __init__(
        self,
        precision: Union[np.float32, np.float64] = np.float32,
        grid_bins: tuple = (224, 224),
        wirelength_weight: float = 1.0,
        congestion_weight: float = 1.0,
        benchmark_name: EDA_BENCH = None,
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
        self.reader = None

        # List of nets, each net is a 2D array of shape
        # [Np, (macro_idx, offset_x, offset_y)]
        self.nets: List[np.ndarray] = None

        # Unlike many other optimization problems where synthetic or 
        # randomly generated data is commonly used, EDA placement tasks 
        # rarely rely on synthetic data. Instead, datasets for EDA problems 
        # are almost exclusively derived from well-established benchmarks.
        # Each benchmark may have its own unique directory structure and 
        # file formats (e.g., ISPD, ICCAD, Bookshelf), which must be parsed 
        # and interpreted according to specific conventions for each dataset.
        # As a result, handling EDA problem data typically requires careful 
        # attention to benchmark-specific formats.
        self.benchmark_name = benchmark_name
        
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
    
    def _check_folder_path(
        self, 
        name: str,
        folder_path: pathlib.Path,
        benchmark_name: EDA_BENCH,
    ):
        # Get valid extensions
        if benchmark_name == EDA_BENCH.ISPD2005:
            valid_exts = [".aux", ".nets", ".nodes", ".pl", ".scl", ".wts"]
        elif benchmark_name == EDA_BENCH.MMS:
            valid_exts = [".aux", ".nets", ".nodes", ".pl", ".scl", ".wts"]
        else:
            raise ValueError(f"Unsupported benchmark name: {benchmark_name}")

        # Check file name prefix consistency and the existence of all required files
        files = os.listdir(folder_path)
        exts_found = []
        for f in files:
            _name, ext = os.path.splitext(os.path.basename(f))
            if _name != name:
                raise ValueError(f"Inconsistent {benchmark_name} file name: {_name}")
            if ext in valid_exts:
                exts_found.append(ext)
        missing = set(valid_exts) - set(exts_found)
        if missing:
            raise FileNotFoundError(
                f"Missing required benchmark files: {sorted(missing)} in {folder_path}"
            )

    def from_ispd2005(
        self, 
        name: str, 
        die: np.ndarray,
        root_path: pathlib.Path
    ): 
        # Check folder path
        folder_path = root_path / name
        self._check_folder_path(
            name=name, 
            folder_path=folder_path, 
            benchmark_name=EDA_BENCH.ISPD2005
        )
        
        # Get data path
        aux_file_path = folder_path / f"{name}.aux"
        nets_file_path = folder_path / f"{name}.nets"
        nodes_file_path = folder_path / f"{name}.nodes"

        # Read data from files
        self.reader = ISPD2005Reader()
        cells, macro_mask = self.reader.from_nodes(str(nodes_file_path))
        cells: np.ndarray
        cells_num: int = cells.shape[0]
        nets = self.reader.from_nets(str(nets_file_path))

        # Call ``from_data`` to set attributes
        self.from_data(
            die=die, cells=cells, cells_num=cells_num, macro_mask=macro_mask, 
            nets=nets, name=name, benchmark_name=EDA_BENCH.ISPD2005
        )
        self.cache["aux"] = aux_file_path
        self.cache["nodes"] = nodes_file_path
        self.cache["result_dir"] = root_path / "dreamplace_results"
        self.cache["result_path"] = root_path / f"dreamplace_results/{name}/{name}.gp.pl"

        # If result path exists, read the result
        if self.cache["result_path"].exists():
            sol = self.reader.from_lg_pl(str(self.cache["result_path"]))
            self.from_data(sol=sol, ref=True)

    def from_mms(
        self,
        name: str,
        die: np.ndarray,
        root_path: pathlib.Path
    ):
        # Check folder path
        folder_path = root_path / name
        self._check_folder_path(
            name=name, 
            folder_path=folder_path, 
            benchmark_name=EDA_BENCH.ISPD2005
        )
        
        # Get data path
        aux_file_path = folder_path / f"{name}.aux"
        nets_file_path = folder_path / f"{name}.nets"
        nodes_file_path = folder_path / f"{name}.nodes"

        # Read data from files
        self.reader = ISPD2005Reader()
        cells, macro_mask = self.reader.from_nodes(str(nodes_file_path))
        cells: np.ndarray
        cells_num: int = cells.shape[0]
        nets = self.reader.from_nets(str(nets_file_path))

        # Call ``from_data`` to set attributes
        self.from_data(
            die=die, cells=cells, cells_num=cells_num, macro_mask=macro_mask, 
            nets=nets, name=name, benchmark_name=EDA_BENCH.MMS
        )
        self.cache["aux"] = aux_file_path
        self.cache["result_dir"] = root_path / "dreamplace_results"
        self.cache["result_path"] = root_path / f"dreamplace_results/{name}/{name}.gp.pl"

        # If result path exists, read the result
        if self.cache["result_path"].exists():
            sol = self.reader.from_lg_pl(str(self.cache["result_path"]))
            self.from_data(sol=sol, ref=True)

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
        benchmark_name: EDA_BENCH = None,
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
        if benchmark_name is not None:
            self.benchmark_name = benchmark_name

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

        # Note: MMS is a special dataset where some macros from the ISPD2005 
        # dataset are released as movable cells. However, MMS does not provide 
        # the coordinate ranges for these macros. The die constraints of ISPD2005   
        # dataset only apply to the small cells, not to these movable macros.  
        # As a result, we cannot perform constraint checking for MMS to ensure 
        # all objects are inside legal regions.

        if self.benchmark_name in [EDA_BENCH.ISPD2005]:
           if check_constr and not self.check_constraints(sol):
                raise ValueError("Invalid solution!")
        
        # Evaluate
        hpwl, congestion_map = self.helper.evaluate(
            sol, self.nets, self.die, self.bin_cols, self.bin_rows
        )
        congestion_map: np.ndarray
        max_congestion = np.max(congestion_map).item()
        avg_congestion = np.mean(congestion_map).item()
        return hpwl, max_congestion, avg_congestion

    def evaluate_w_gap(self, check_constr: bool = True):
        info = (
            "Due to the particularity of EDA problems, "
            "which involve multiple evaluation metrics, "
            "we do not recommend using gap as an evaluation method."
        )
        raise NotImplementedError(info)
   