
r"""
GNN4CO Beam Decoder.
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
from ml4co_kit.solver.lib.beam.mcl_beam import mcl_beam
from ml4co_kit.solver.lib.beam.mis_beam import mis_beam
from .base import GNN4CODecoder


class GNN4COBeamDecoder(GNN4CODecoder):
    def __init__(
        self, 
        sparse_factor: int, 
        beam_size: int = 16
    ) -> None:
        super(GNN4COBeamDecoder, self).__init__(sparse_factor)
        self.beam_size = beam_size

    def _decode(self, task_data: TaskBase):
        # Get task type
        task_type = task_data.task_type

        # Decode according to task type
        if task_type == TASK_TYPE.MCL:
            mcl_beam(task_data)
        elif task_type == TASK_TYPE.MIS:
            mis_beam(task_data)
        else:
            raise NotImplementedError()