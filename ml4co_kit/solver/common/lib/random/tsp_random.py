r"""
Random Initialization Algorithm for TSP
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
from ml4co_kit.task.routing.tsp import TSPTask


def _tsp_random(nodes_num: int) -> np.ndarray:
    # Random index
    index = np.arange(1, nodes_num)
    np.random.shuffle(index)

    # Random tour
    random_tour = np.insert(index, [0, len(index)], [0, 0])
    return random_tour


def tsp_random(task_data: TSPTask):
    # Call ``_tsp_random`` to get the random tour
    random_tour = _tsp_random(nodes_num=task_data.nodes_num)

    # Store the random tour in the task_data
    task_data.from_data(sol=random_tour, ref=False)