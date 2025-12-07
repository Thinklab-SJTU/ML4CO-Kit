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
                self.ref_sol = sol.astype(self.precision)
            else:
                self.sol = sol.astype(self.precision)
        if name is not None:
            self.name = name
