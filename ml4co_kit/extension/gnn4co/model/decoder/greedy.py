
r"""
GNN4CO Greedy Decoder.
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


from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.greedy.tsp_greedy import tsp_greedy
from ml4co_kit.solver.lib.greedy.mcl_greedy import mcl_greedy
from ml4co_kit.solver.lib.greedy.mis_greedy import mis_greedy
from ml4co_kit.solver.lib.greedy.mvc_greedy import mvc_greedy
from ml4co_kit.solver.lib.greedy.mcut_greedy import mcut_greedy
from ml4co_kit.solver.lib.greedy.atsp_greedy import atsp_greedy
from .base import GNN4CODecoder


class GNN4COGreedyDecoder(GNN4CODecoder):
    def __init__(self, sparse_factor: int) -> None:
        super(GNN4COGreedyDecoder, self).__init__(sparse_factor)

    def _decode(self, task_data: TaskBase):
        # Get task type
        task_type = task_data.task_type

        # Decode according to task type
        if task_type == TASK_TYPE.TSP:
            tsp_greedy(task_data)
        elif task_type == TASK_TYPE.ATSP:
            atsp_greedy(task_data)
        elif task_type == TASK_TYPE.MCL:
            mcl_greedy(task_data)
        elif task_type == TASK_TYPE.MCUT:
            mcut_greedy(task_data)
        elif task_type == TASK_TYPE.MIS:
            mis_greedy(task_data)
        elif task_type == TASK_TYPE.MVC:
            mvc_greedy(task_data)
        else:
            raise NotImplementedError()