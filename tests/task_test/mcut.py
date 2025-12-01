r"""
MCut Task Tester.
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


import os
import pathlib
from ml4co_kit import MCutTask
from tests.task_test.base import TaskTesterBase


class MCutTaskTester(TaskTesterBase):
    def __init__(self):
        super(MCutTaskTester, self).__init__(
            test_task_class=MCutTask,
            pickle_files_list=[
                pathlib.Path("test_dataset/mcut/task/mcut_ba-large_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/mcut/task/mcut_ba-small_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/mcut/task/mcut_ba-small_uniform-weighted_task.pkl"),
            ],
        )
        
    def _test_other_rw_methods(self):
        
        ##################################################
        #    Test-1: Read data from adj_matrix_weighed   #
        ##################################################
        
        # 1.1 Transfer pickle to adj_matrix
        task = MCutTask(edge_weighted=True)
        task.from_pickle("test_dataset/mcut/task/mcut_ba-small_uniform-weighted_task.pkl")
        adj_matrix_weighted = task.to_adj_matrix(with_edge_weights=True)
        
        # 1.2 Read data from adj_matrix
        new_task = MCutTask(edge_weighted=True)
        new_task.from_adj_matrix_weighted(adj_matrix_weighted=adj_matrix_weighted)

        # 1.3 Verify the consistency
        task_edge_index = task.edge_index
        task_edges_weight = task.edges_weight
        new_task_edge_index = new_task.edge_index
        new_task_edges_weight = new_task.edges_weight
        if (task_edge_index != new_task_edge_index).any():
            raise ValueError("Inconsistent data using `adj_matrix_weighted`.")
        if (task_edges_weight != new_task_edges_weight).any():
            raise ValueError("Inconsistent data using `adj_matrix_weighted`.")
    
    def _test_render(self):
        # Read data
        task = MCutTask()
        task.from_pickle("test_dataset/mcut/task/mcut_ba-small_no-weighted_task.pkl")
        task.sol = task.ref_sol
        
        # Render (problem)
        tmp_path = self._make_tmp_file()
        task.render(save_path=pathlib.Path(tmp_path + ".png"), with_sol=False)
        
        # Render (solution)
        task.render(save_path=pathlib.Path(tmp_path + "_sol.png"), with_sol=True)
        
        # Clean up
        os.remove(tmp_path + ".png")
        os.remove(tmp_path + "_sol.png")