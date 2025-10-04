r"""
Base class for all generators in the ML4CO kit.
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
from typing import Union, Any
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class GeneratorBase(object):
    """Base class for all generators."""

    def __init__(
        self, 
        task_type: TASK_TYPE, 
        distribution_type: Any,
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        self.task_type = task_type
        self.distribution_type = distribution_type
        self.precision = precision
        self.generate_func_dict: dict = None
        
    def generate(self) -> TaskBase:
        if self.distribution_type not in self.generate_func_dict:
            raise NotImplementedError(
                f"The distribution type {self.distribution_type} is not supported."
            )
        else:
            return self.generate_func_dict[self.distribution_type]()
        
    def __repr__(self) -> str:
        return f"{self.task_type.value}Generator"