r"""
RLSA Optimizer Tester.
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


from ml4co_kit import *
from ml4co_kit.extension.gnn4co import (
    GNN4COModel, GNN4COEnv, GNNEncoder, GNN4COGreedyDecoder
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
    decoder=GNN4COGreedyDecoder(sparse_factor=1),
    weight_path="weights/gnn4co_mcl_rb-small_sparse.pt"
)

# Test on MCut (sparse)
gnn4mcut_model = GNN4COModel(
    env=GNN4COEnv(
        task_type=TASK_TYPE.MCUT, 
        wrapper=MCutWrapper(), 
        mode="solve", 
        sparse_factor=1, 
        device="cpu"
    ),
    encoder=GNNEncoder(
        task_type=TASK_TYPE.MCUT, 
        sparse=True, 
        block_layers=[2,4,4,2]
    ),
    decoder=GNN4COGreedyDecoder(sparse_factor=1),
    weight_path="weights/gnn4co_mcut_ba-small_sparse.pt"
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
    decoder=GNN4COGreedyDecoder(sparse_factor=1),
    weight_path="weights/gnn4co_mis_satlib_sparse.pt"
)

# Test on MVC (sparse)
gnn4mvc_model = GNN4COModel(
    env=GNN4COEnv(
        task_type=TASK_TYPE.MVC, 
        wrapper=MVCWrapper(), 
        mode="solve", 
        sparse_factor=1, 
        device="cpu"
    ),
    encoder=GNNEncoder(
        task_type=TASK_TYPE.MVC, 
        sparse=True, 
        block_layers=[2,4,4,2]
    ),
    decoder=GNN4COGreedyDecoder(sparse_factor=1),
    weight_path="weights/gnn4co_mvc_rb-small_sparse.pt"
)

# Optimizers
optimizer_torch = RLSAOptimizer(impl_type=IMPL_TYPE.TORCH)


class RLSAOptimizerTester(SolverTesterBase):
    def __init__(self, device: str = "cpu"):
        super(RLSAOptimizerTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=GNN4COSolver,
            test_task_type_list=[
                TASK_TYPE.MCL, 
                TASK_TYPE.MCUT, 
                TASK_TYPE.MIS, 
                TASK_TYPE.MVC, 
            ],
            test_args_list=[
                # MCl (sparse)
                {"model": gnn4mcl_model, "device": device, "optimizer": optimizer_torch},
                # MCut (sparse)
                {"model": gnn4mcut_model, "device": device, "optimizer": optimizer_torch},
                # MIS (sparse)
                {"model": gnn4mis_model, "device": device, "optimizer": optimizer_torch},
                # MVC (sparse)
                {"model": gnn4mvc_model, "device": device, "optimizer": optimizer_torch},
            ],
            exclude_test_files_list=[
                [], # MCl (sparse)
                [], # MCut (sparse)
                [], # MIS (sparse)
                [], # MVC (sparse)
            ],
            info="RLSA Optimizer"
        )
        
    def pre_test(self):
        pass