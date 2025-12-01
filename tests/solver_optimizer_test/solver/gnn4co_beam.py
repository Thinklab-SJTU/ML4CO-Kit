r"""
GNN4CO Solver Tester (Beam).
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


import pathlib
from ml4co_kit import *
from ml4co_kit.extension.gnn4co import (
    GNN4COModel, GNN4COEnv, GNNEncoder, GNN4COBeamDecoder
)
from tests.solver_optimizer_test.base import SolverTesterBase


# Test on MCl (sparse)
gnn4mcl_model = GNN4COModel(
    env=GNN4COEnv(
        task_type=TASK_TYPE.MCL, 
        wrapper=MClWrapper(), 
        mode="solve", 
        sparse_factor=1, 
        device="cpu"
    ),
    encoder=GNNEncoder(
        task_type=TASK_TYPE.MCL, 
        sparse=True, 
        block_layers=[2,4,4,2]
    ),
    decoder=GNN4COBeamDecoder(sparse_factor=1, beam_size=16),
    weight_path="weights/gnn4co_mcl_rb-small_sparse.pt"
)

# Test on MIS (sparse)
gnn4mis_model = GNN4COModel(
    env=GNN4COEnv(
        task_type=TASK_TYPE.MIS, 
        wrapper=MISWrapper(), 
        mode="solve", 
        sparse_factor=1, 
        device="cpu"
    ),
    encoder=GNNEncoder(
        task_type=TASK_TYPE.MIS, 
        sparse=True, 
        block_layers=[2,4,4,2]
    ),
    decoder=GNN4COBeamDecoder(sparse_factor=1, beam_size=16),
    weight_path="weights/gnn4co_mis_er-700-800_sparse.pt"
)


class GNN4COBeamSolverTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(GNN4COBeamSolverTester, self).__init__(
            mode_list=["solve", "batch_solve"],
            test_solver_class=GNN4COSolver,
            test_task_type_list=[
                TASK_TYPE.MCL, 
                TASK_TYPE.MIS, 
            ],
            test_args_list=[
                # MCl (sparse)
                {"model": gnn4mcl_model, "device": device},
                # MIS (sparse)
                {"model": gnn4mis_model, "device": device},
            ],
            exclude_test_files_list=[
                [], # MCl (sparse)
                [], # MIS (sparse)
            ],
            info="Beam Decoding"
        )
        
    def pre_test(self):
        pass