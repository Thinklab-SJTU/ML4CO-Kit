r"""
Embedder Module.
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


from .atsp import ATSPEmbedder
from .graph import GraphEmbedder
from .tsp import TSPEmbedder
from ml4co_kit.task.base import TASK_TYPE


EMBEDDER_DICT = {
<<<<<<< HEAD
    "ATSP": ATSPEmbedder,
    "MCl": MClEmbedder,
    "MIS": MISEmbedder,
    "MCut": MCutEmbedder,
    "MVC": MVCEmbedder,
    "TSP": TSPEmbedder,
=======
    TASK_TYPE.ATSP: ATSPEmbedder,
    TASK_TYPE.TSP: TSPEmbedder,
    TASK_TYPE.MCL: GraphEmbedder,
    TASK_TYPE.MIS: GraphEmbedder,
    TASK_TYPE.MCUT: GraphEmbedder,
    TASK_TYPE.MVC: GraphEmbedder,
>>>>>>> upstream/main
}


def get_embedder_by_task(task_type: TASK_TYPE):
    return EMBEDDER_DICT[task_type]