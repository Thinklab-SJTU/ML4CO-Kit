r"""
Standard Cell Placement task.
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
from typing import Union, List, Tuple
from ml4co_kit.task.base import TASK_TYPE
from .base import PlacementTask


class StandardCellPlacementTask(PlacementTask):
    """
    Standard Cell Placement Task.
    """
    
    def __init__(
        self,
        precision: Union[np.float32, np.float64] = np.float32,
        w_hpwl: float = 1.0,
        w_density: float = 1000.0,
        w_congestion: float = 500.0,
        grid_size: int = 32,
        target_density: float = 0.8
    ):
        super(StandardCellPlacementTask, self).__init__(
            task_type=TASK_TYPE.STANDARD_CELL_PLACEMENT,
            minimize=True,
            precision=precision,
            w_hpwl=w_hpwl,
            w_overlap=0.0,
            w_oob=0.0
        )
        self.w_density = w_density
        self.w_congestion = w_congestion
        self.grid_size = grid_size
        self.target_density = target_density
        self.fixed_macros: List[dict] = []
        self.placement_rows: List[dict] = []
    
    def from_data(
        self, 
        canvas_width: float = None,
        canvas_height: float = None,
        macros: List[dict] = None,
        std_cells: List[dict] = None,
        nets: List[dict] = None,
        sol: np.ndarray = None,
        ref: bool = False,
        name: str = None,
        fixed_macros: List[dict] = None,
        placement_rows: List[dict] = None
    ):
        super().from_data(canvas_width, canvas_height, macros, std_cells, nets, sol, ref, name)
        
        if fixed_macros is not None:
            self.fixed_macros = fixed_macros
        if placement_rows is not None:
            self.placement_rows = placement_rows

    def _check_sol_dim(self):
        if self.sol.ndim != 2 or self.sol.shape[1] != 2:
            raise ValueError("Solution shape should be [n_std_cells, 2]")
        if self.sol.shape[0] != self.std_cells_num:
            raise ValueError(f"Expected {self.std_cells_num} standard cells, got {self.sol.shape[0]}")

    def _check_ref_sol_dim(self):
        if self.ref_sol.ndim != 2 or self.ref_sol.shape[1] != 2:
            raise ValueError("Reference solution shape should be [n_std_cells, 2]")
        if self.ref_sol.shape[0] != self.std_cells_num:
            raise ValueError(f"Expected {self.std_cells_num} standard cells, got {self.ref_sol.shape[0]}")

    def _compute_hpwl_for_std_cells(self, sol: np.ndarray) -> float:
        total_hpwl = 0.0
        for net in self.nets:
            cell_indices = net.get("cells", [])
            if len(cell_indices) == 0:
                continue
            
            pin_positions = []
            for cell_idx in cell_indices:
                if cell_idx >= len(sol):
                    continue
                cell_center = sol[cell_idx]
                pin_offset = net.get("pin_offsets", {}).get(cell_idx, [0.0, 0.0])
                pin_pos = cell_center + np.array(pin_offset)
                pin_positions.append(pin_pos)
            
            if len(pin_positions) > 0:
                pin_positions = np.array(pin_positions)
                bbox_min = pin_positions.min(axis=0)
                bbox_max = pin_positions.max(axis=0)
                hpwl = (bbox_max[0] - bbox_min[0]) + (bbox_max[1] - bbox_min[1])
                total_hpwl += hpwl
        
        return float(total_hpwl)

    def _compute_density_map(self, sol: np.ndarray) -> np.ndarray:
        grid_w = self.canvas_width / self.grid_size
        grid_h = self.canvas_height / self.grid_size
        
        density_map = np.zeros((self.grid_size, self.grid_size), dtype=self.precision)
        
        for i in range(len(self.fixed_macros)):
            macro = self.fixed_macros[i]
            x, y = macro["x"], macro["y"]
            w, h = macro["width"], macro["height"]
            
            x_min, x_max = x - w/2, x + w/2
            y_min, y_max = y - h/2, y + h/2
            
            gx_min = max(0, int(x_min / grid_w))
            gx_max = min(self.grid_size - 1, int(x_max / grid_w))
            gy_min = max(0, int(y_min / grid_h))
            gy_max = min(self.grid_size - 1, int(y_max / grid_h))
            
            for gy in range(gy_min, gy_max + 1):
                for gx in range(gx_min, gx_max + 1):
                    density_map[gy, gx] += w * h / ((gx_max - gx_min + 1) * (gy_max - gy_min + 1))
        
        for i in range(self.std_cells_num):
            w_i = self.std_cells[i]["width"]
            h_i = self.std_cells[i]["height"]
            x_i, y_i = sol[i]
            
            x_min, x_max = x_i - w_i/2, x_i + w_i/2
            y_min, y_max = y_i - h_i/2, y_i + h_i/2
            
            gx_min = max(0, int(x_min / grid_w))
            gx_max = min(self.grid_size - 1, int(x_max / grid_w))
            gy_min = max(0, int(y_min / grid_h))
            gy_max = min(self.grid_size - 1, int(y_max / grid_h))
            
            for gy in range(gy_min, gy_max + 1):
                for gx in range(gx_min, gx_max + 1):
                    cell_area = w_i * h_i
                    num_grids = (gx_max - gx_min + 1) * (gy_max - gy_min + 1)
                    density_map[gy, gx] += cell_area / num_grids
        
        return density_map

    def _compute_density_overflow(self, sol: np.ndarray) -> float:
        if self.canvas_width == 0 or self.canvas_height == 0:
            return 0.0
        
        density_map = self._compute_density_map(sol)
        
        grid_area = (self.canvas_width / self.grid_size) * (self.canvas_height / self.grid_size)
        target_area = grid_area * self.target_density
        
        overflow = np.sum(np.maximum(0, density_map - target_area))
        
        return float(overflow)
    
    def _compute_congestion(self, sol: np.ndarray) -> float:
        if self.canvas_width == 0 or self.canvas_height == 0:
            return 0.0
        
        grid_w = self.canvas_width / self.grid_size
        grid_h = self.canvas_height / self.grid_size
        
        h_demand = np.zeros((self.grid_size, self.grid_size - 1), dtype=self.precision)
        v_demand = np.zeros((self.grid_size - 1, self.grid_size), dtype=self.precision)
        
        for net in self.nets:
            cell_indices = net.get("cells", [])
            if len(cell_indices) < 2:
                continue
            
            positions = []
            for cell_idx in cell_indices:
                if cell_idx >= len(sol):
                    continue
                positions.append(sol[cell_idx])
            
            if len(positions) < 2:
                continue
            
            positions = np.array(positions)
            x_min, y_min = positions.min(axis=0)
            x_max, y_max = positions.max(axis=0)
            
            bbox_hpwl = (x_max - x_min) + (y_max - y_min)
            
            gx_min = max(0, int(x_min / grid_w))
            gx_max = min(self.grid_size - 1, int(x_max / grid_w))
            gy_min = max(0, int(y_min / grid_h))
            gy_max = min(self.grid_size - 1, int(y_max / grid_h))
            
            for gy in range(gy_min, gy_max + 1):
                for gx in range(gx_min, gx_max):
                    h_demand[gy, gx] += 1
            
            for gy in range(gy_min, gy_max):
                for gx in range(gx_min, gx_max + 1):
                    v_demand[gy, gx] += 1
        
        total_congestion = np.sum(h_demand) + np.sum(v_demand)
        return float(total_congestion)

    def evaluate(self, sol: np.ndarray = None, ref: bool = False) -> float:
        if sol is None:
            if ref:
                self._check_ref_sol_not_none()
                sol = self.ref_sol
            else:
                self._check_sol_not_none()
                sol = self.sol
        
        hpwl = self._compute_hpwl_for_std_cells(sol)
        density_overflow = self._compute_density_overflow(sol)
        congestion = self._compute_congestion(sol)
        
        cost = self.w_hpwl * hpwl + self.w_density * density_overflow + self.w_congestion * congestion
        return float(cost)

    def check_constraints(self, sol: np.ndarray) -> bool:
        density_overflow = self._compute_density_overflow(sol)
        
        for i in range(self.std_cells_num):
            x_i, y_i = sol[i]
            w_i = self.std_cells[i]["width"]
            h_i = self.std_cells[i]["height"]
            
            for macro in self.fixed_macros:
                x_m, y_m = macro["x"], macro["y"]
                w_m, h_m = macro["width"], macro["height"]
                
                x1_min, x1_max = x_i - w_i/2, x_i + w_i/2
                y1_min, y1_max = y_i - h_i/2, y_i + h_i/2
                x2_min, x2_max = x_m - w_m/2, x_m + w_m/2
                y2_min, y2_max = y_m - h_m/2, y_m + h_m/2
                
                overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                
                if overlap_x > 0 and overlap_y > 0:
                    return False
        
        return density_overflow == 0.0
