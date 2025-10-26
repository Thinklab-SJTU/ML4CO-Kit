r"""
SAT Greedy Solver Tester.

Note: This tester is currently disabled because GNN4CO doesn't have SAT embedder 
implemented yet. The file is preserved for future implementation.

To enable this tester:
1. Add SAT embedder to ml4co_kit/extension/gnn4co/model/embedder/
2. Register it in the EMBEDDER_DICT
3. Uncomment the implementation below
4. Add it back to test_solver_optimizer.py

See docs/SAT_Solver_Greedy_Status.md for detailed instructions.
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


from ml4co_kit import TASK_TYPE
from tests.solver_optimizer_test.base import SolverTesterBase


class SATGreedySolverTester(SolverTesterBase):
    """
    SAT Greedy Solver Tester (Currently Disabled)
    
    This tester is preserved for future implementation when SAT embedder
    is added to GNN4CO. The SAT greedy solver uses heuristic methods and
    doesn't require neural networks, but the GreedySolver interface
    requires a GNN4CO model parameter.
    """
    
    def __init__(self, device: str = "cpu"):
        raise NotImplementedError(
            "SATGreedySolverTester is currently disabled. "
            "Reason: GNN4CO doesn't have SAT embedder implemented yet. "
            "See docs/SAT_Solver_Greedy_Status.md for details."
        )
        
        # TODO: Uncomment when SAT embedder is available
        # from ml4co_kit import GreedySolver
        # from ml4co_kit.extension.gnn4co import GNN4COModel, GNN4COEnv, GNNEncoder
        # 
        # gnn4sat_model = GNN4COModel(
        #     env=GNN4COEnv(task="SAT", mode="solve", sparse_factor=1, device=device),
        #     encoder=GNNEncoder(task="SAT", sparse=True, block_layers=[2,4,4,2]),
        #     weight_path=None  # Heuristic-based, no weights needed
        # )
        # 
        # super(SATGreedySolverTester, self).__init__(
        #     mode_list=["solve"],
        #     test_solver_class=GreedySolver,
        #     test_task_type_list=[TASK_TYPE.SAT],
        #     test_args_list=[{"model": gnn4sat_model, "device": device}],
        #     exclude_test_files_list=[[]]
        # )
        
    def pre_test(self):
        """Pre-test setup (currently not used)."""
        pass