r"""
Base class for Routing tasks.
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
from .base import EDATaskBase


class RoutingTask(EDATaskBase):
    
    def __init__(
        self,
        task_type: TASK_TYPE,
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(RoutingTask, self).__init__(task_type, minimize, precision)

    def _check_sol_dim(self):
        raise NotImplementedError

    def _check_ref_sol_dim(self):
        raise NotImplementedError

    def evaluate(self, sol: np.ndarray = None, ref: bool = False):
        raise NotImplementedError


class GlobalRoutingTask(RoutingTask):
    """
    Global Routing Task.
    """
    
    def __init__(
        self,
        precision: Union[np.float32, np.float64] = np.float32,
        w_wirelength: float = 1.0,
        w_overflow: float = 1000.0
    ):
        super(GlobalRoutingTask, self).__init__(
            task_type=TASK_TYPE.GLOBAL_ROUTING,
            minimize=True,
            precision=precision
        )
        self.w_wirelength = w_wirelength
        self.w_overflow = w_overflow
        self.grid_width: int = 0
        self.grid_height: int = 0
        self.h_capacity: np.ndarray = None
        self.v_capacity: np.ndarray = None

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
        grid_width: int = None,
        grid_height: int = None,
        h_capacity: np.ndarray = None,
        v_capacity: np.ndarray = None
    ):
        super().from_data(canvas_width, canvas_height, macros, std_cells, nets, sol, ref, name)
        
        if grid_width is not None:
            self.grid_width = grid_width
        if grid_height is not None:
            self.grid_height = grid_height
        if h_capacity is not None:
            self.h_capacity = h_capacity
        if v_capacity is not None:
            self.v_capacity = v_capacity

    def _check_sol_dim(self):
        if self.sol.ndim != 3:
            raise ValueError("Solution should be 3D array [2, H, W]")
        if self.sol.shape[0] != 2:
            raise ValueError("First dimension should be 2 (Horizontal, Vertical)")
        if self.sol.shape[1] != self.grid_height:
            raise ValueError(f"Expected height {self.grid_height}, got {self.sol.shape[1]}")
        if self.sol.shape[2] != self.grid_width:
            raise ValueError(f"Expected width {self.grid_width}, got {self.sol.shape[2]}")

    def _check_ref_sol_dim(self):
        if self.ref_sol.ndim != 3:
            raise ValueError("Reference solution should be 3D array [2, H, W]")
        if self.ref_sol.shape[0] != 2:
            raise ValueError("First dimension should be 2 (Horizontal, Vertical)")
        if self.ref_sol.shape[1] != self.grid_height:
            raise ValueError(f"Expected height {self.grid_height}, got {self.ref_sol.shape[1]}")
        if self.ref_sol.shape[2] != self.grid_width:
            raise ValueError(f"Expected width {self.grid_width}, got {self.ref_sol.shape[2]}")

    def _compute_wirelength(self, sol: np.ndarray) -> float:
        return float(np.sum(sol))

    def _compute_overflow(self, sol: np.ndarray) -> float:
        total_overflow = 0.0
        
        h_usage = sol[0, :, :-1]
        if self.h_capacity is not None:
            if np.isscalar(self.h_capacity):
                diff = h_usage - self.h_capacity
            else:
                diff = h_usage - self.h_capacity
            total_overflow += np.sum(np.maximum(0, diff))
            
        v_usage = sol[1, :-1, :]
        if self.v_capacity is not None:
            if np.isscalar(self.v_capacity):
                diff = v_usage - self.v_capacity
            else:
                diff = v_usage - self.v_capacity
            total_overflow += np.sum(np.maximum(0, diff))
            
        return float(total_overflow)

    def evaluate(self, sol: np.ndarray = None, ref: bool = False) -> float:
        if sol is None:
            if ref:
                self._check_ref_sol_not_none()
                sol = self.ref_sol
            else:
                self._check_sol_not_none()
                sol = self.sol
        
        wl = self._compute_wirelength(sol)
        overflow = self._compute_overflow(sol)
        
        cost = self.w_wirelength * wl + self.w_overflow * overflow
        return float(cost)

    def check_constraints(self, sol: np.ndarray) -> bool:
        overflow = self._compute_overflow(sol)
        return overflow == 0.0

    def get_grid_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        if self.canvas_width == 0 or self.canvas_height == 0:
            raise ValueError("Canvas size not set")
        
        gx = int((x / self.canvas_width) * self.grid_width)
        gy = int((y / self.canvas_height) * self.grid_height)
        
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        
        return gx, gy
