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
