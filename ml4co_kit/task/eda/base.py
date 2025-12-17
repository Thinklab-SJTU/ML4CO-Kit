r"""
Base class for EDA tasks.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import Union, List
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class EDATaskBase(TaskBase):
    
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(EDATaskBase, self).__init__(task_type, minimize, precision)
        self.canvas_width: float = 0.0
        self.canvas_height: float = 0.0
        self.macros: List[dict] = []
        self.macros_num: int = 0
        self.std_cells: List[dict] = [] # Standard cells (optional)
        self.std_cells_num: int = 0
        self.nets: List[dict] = []
        self.nets_num: int = 0

    def from_data(
        self, 
        canvas_width: float = None,
        canvas_height: float = None,
        macros: List[dict] = None,
        std_cells: List[dict] = None,
        nets: List[dict] = None,
        sol: np.ndarray = None,
        ref: bool = False,
        name: str = None
    ):
        if canvas_width is not None:
            self.canvas_width = canvas_width
        if canvas_height is not None:
            self.canvas_height = canvas_height
        if macros is not None:
            self.macros = macros
            self.macros_num = len(macros)
        if std_cells is not None:
            self.std_cells = std_cells
            self.std_cells_num = len(std_cells)
        if nets is not None:
            self.nets = nets
            self.nets_num = len(nets)
        
        if sol is not None:
            if ref:
                self.ref_sol = sol
                self._check_ref_sol_dim()
            else:
                self.sol = sol
                self._check_sol_dim()
        
        if name is not None:
            self.name = name

    def _check_sol_dim(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _check_ref_sol_dim(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def render(self):
        raise NotImplementedError("Subclasses should implement this method.")


class PlacementTask(EDATaskBase):
    
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32,
        w_hpwl: float = 1.0,
        w_overlap: float = 1000.0,
        w_oob: float = 1000.0
    ):
        super(PlacementTask, self).__init__(task_type, minimize, precision)
        self.w_hpwl = w_hpwl
        self.w_overlap = w_overlap
        self.w_oob = w_oob

    def _check_sol_dim(self):
        if self.sol.ndim != 2 or self.sol.shape[1] != 2:
            raise ValueError("Solution shape should be [n_macros, 2]")
        if self.sol.shape[0] != self.macros_num:
            raise ValueError(f"Expected {self.macros_num} macros, got {self.sol.shape[0]}")

    def _check_ref_sol_dim(self):
        if self.ref_sol.ndim != 2 or self.ref_sol.shape[1] != 2:
            raise ValueError("Reference solution shape should be [n_macros, 2]")
        if self.ref_sol.shape[0] != self.macros_num:
            raise ValueError(f"Expected {self.macros_num} macros, got {self.ref_sol.shape[0]}")

    def _compute_hpwl(self, sol: np.ndarray) -> float:
        """
        Compute Half-Perimeter Wirelength (HPWL).
        
        Args:
            sol: Solution array of shape [n_macros, 2]. 
                 Represents the CENTER coordinates (x, y) of macros.
        """
        total_hpwl = 0.0
        for net in self.nets:
            macro_indices = net["macros"]
            if len(macro_indices) == 0:
                continue
            
            pin_positions = []
            for macro_idx in macro_indices:
                # macro_center is directly sol[macro_idx]
                macro_center = sol[macro_idx]
                pin_offset = net.get("pin_offsets", {}).get(macro_idx, [0.0, 0.0])
                pin_pos = macro_center + np.array(pin_offset)
                pin_positions.append(pin_pos)
            
            pin_positions = np.array(pin_positions)
            bbox_min = pin_positions.min(axis=0)
            bbox_max = pin_positions.max(axis=0)
            hpwl = (bbox_max[0] - bbox_min[0]) + (bbox_max[1] - bbox_min[1])
            total_hpwl += hpwl
        
        return float(total_hpwl)

    def _compute_overlap(self, sol: np.ndarray) -> float:
        """
        Compute total overlap area.
        
        Args:
            sol: Solution array of shape [n_macros, 2]. 
                 Represents the CENTER coordinates (x, y) of macros.
        """
        total_overlap = 0.0
        for i in range(self.macros_num):
            w_i = self.macros[i]["width"]
            h_i = self.macros[i]["height"]
            x_i, y_i = sol[i]
            
            # Bounding box of macro i
            x1_min = x_i - w_i / 2
            x1_max = x_i + w_i / 2
            y1_min = y_i - h_i / 2
            y1_max = y_i + h_i / 2
            
            for j in range(i + 1, self.macros_num):
                w_j = self.macros[j]["width"]
                h_j = self.macros[j]["height"]
                x_j, y_j = sol[j]
                
                # Bounding box of macro j
                x2_min = x_j - w_j / 2
                x2_max = x_j + w_j / 2
                y2_min = y_j - h_j / 2
                y2_max = y_j + h_j / 2
                
                overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                total_overlap += overlap_x * overlap_y
        
        return float(total_overlap)

    def _compute_oob(self, sol: np.ndarray) -> float:
        """
        Compute Out-Of-Bounds (OOB) area.
        
        Args:
            sol: Solution array of shape [n_macros, 2]. 
                 Represents the CENTER coordinates (x, y) of macros.
        """
        total_oob = 0.0
        for i in range(self.macros_num):
            w_i = self.macros[i]["width"]
            h_i = self.macros[i]["height"]
            x_i, y_i = sol[i]
            
            # Bounding box of macro i
            x_min = x_i - w_i / 2
            x_max = x_i + w_i / 2
            y_min = y_i - h_i / 2
            y_max = y_i + h_i / 2
            
            oob_left = max(0, -x_min)
            oob_right = max(0, x_max - self.canvas_width)
            oob_bottom = max(0, -y_min)
            oob_top = max(0, y_max - self.canvas_height)
            
            if oob_left > 0 or oob_right > 0:
                total_oob += (oob_left + oob_right) * h_i
            
            if oob_bottom > 0 or oob_top > 0:
                width_inside = min(x_max, self.canvas_width) - max(x_min, 0)
                if width_inside > 0:
                    total_oob += (oob_bottom + oob_top) * width_inside
                else:
                    total_oob += (oob_bottom + oob_top) * w_i
        
        return float(total_oob)
