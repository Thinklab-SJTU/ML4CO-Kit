r"""
Macro Placement task.
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
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from .base import PlacementTask


class MacroPlacementTask(PlacementTask):
    
    def __init__(
        self,
        precision: Union[np.float32, np.float64] = np.float32,
        w_hpwl: float = 1.0,
        w_overlap: float = 1000.0,
        w_oob: float = 1000.0
    ):
        super(MacroPlacementTask, self).__init__(
            task_type=TASK_TYPE.MACRO_PLACEMENT,
            minimize=True,
            precision=precision,
            w_hpwl=w_hpwl,
            w_overlap=w_overlap,
            w_oob=w_oob
        )

    def evaluate(self, sol: np.ndarray = None, ref: bool = False) -> float:
        """
        Evaluate the solution.
        Cost = w_hpwl * HPWL + w_overlap * Overlap + w_oob * OOB
        """
        if sol is None:
            if ref:
                self._check_ref_sol_not_none()
                sol = self.ref_sol
            else:
                self._check_sol_not_none()
                sol = self.sol
        
        hpwl = self._compute_hpwl(sol)
        overlap = self._compute_overlap(sol)
        oob = self._compute_oob(sol)
        
        cost = self.w_hpwl * hpwl + self.w_overlap * overlap + self.w_oob * oob
        return float(cost)

    def check_constraints(self, sol: np.ndarray) -> bool:
        """
        Check if the solution satisfies constraints (no overlap, no OOB).
        """
        overlap = self._compute_overlap(sol)
        oob = self._compute_oob(sol)
        return overlap == 0.0 and oob == 0.0
