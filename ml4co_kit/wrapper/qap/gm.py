r"""
GM Wrapper.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import Union, List
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.gm import GMTask
from ml4co_kit.wrapper.base import WrapperBase


class GMWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(GMWrapper, self).__init__(
            task_type=TASK_TYPE.GM, precision=precision
        )
        self.task_list: List[GMTask] = list()