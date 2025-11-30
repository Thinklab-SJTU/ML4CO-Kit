
r"""
GNN4CO MCTS Decoder.
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
from ml4co_kit.solver.lib.mcts.tsp_mcts import tsp_mcts
from .base import GNN4CODecoder


class GNN4COMCTSDecoder(GNN4CODecoder):
    def __init__(
        self, 
        sparse_factor: int, 
        mcts_time_limit: float = 1.0,
        mcts_max_depth: int = 10, 
        mcts_type_2opt: int = 1, 
        mcts_max_iterations_2opt: int = 5000,
    ) -> None:
        super(GNN4COMCTSDecoder, self).__init__(sparse_factor)
        self.mcts_time_limit = mcts_time_limit
        self.mcts_max_depth = mcts_max_depth
        self.mcts_type_2opt = mcts_type_2opt
        self.mcts_max_iterations_2opt = mcts_max_iterations_2opt

    def _decode(self, task_data: TaskBase):
        # Get task type
        task_type = task_data.task_type

        # Decode according to task type
        if task_type == TASK_TYPE.TSP:
            tsp_mcts(task_data)
        else:
            raise NotImplementedError()