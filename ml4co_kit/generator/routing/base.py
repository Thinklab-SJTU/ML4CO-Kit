r"""
Base Class for Routing Problem Generators.
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
from typing import Union, Any
from ml4co_kit.generator.base import GeneratorBase, TASK_TYPE


class RoutingGeneratorBase(GeneratorBase):
    """Base class for routing problem generators."""

    def __init__(
        self, 
        task_type: TASK_TYPE, 
        distribution_type: Any,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        # Super Initialization
        super(RoutingGeneratorBase, self).__init__(
            task_type=task_type, 
            distribution_type=distribution_type,
            precision=precision
        )