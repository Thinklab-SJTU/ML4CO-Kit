

r"""
Graph Edit Distance (GED) Generator.
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
import networkx as nx
from enum import Enum
from typing import Union
from ml4co_kit.task.qap.ged import GEDTask
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.base import QAPGraphBase
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.generator.qap.base import QAPGraphGenerator


class GED_TYPE(str, Enum):
    """Define the graph types as an enumeration."""
    # TODO
    pass


class GEDGenerator(GeneratorBase):
    """Generate GED Tasks."""
    # TODO
    pass
    