r"""
MIS Task Tester.
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
from ml4co_kit import MISTask
from tests.task_test.base import TaskTesterBase


class MISTaskTester(TaskTesterBase):
    def __init__(self):
        super(MISTaskTester, self).__init__(
            test_task_class=MISTask,
            pickle_files_list=[
                pathlib.Path("test_dataset/mis/task/mis_er-700-800_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/mis/task/mis_rb-small_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/mis/task/mis_rb-small_uniform-weighted_task.pkl"),
                pathlib.Path("test_dataset/mis/task/mis_satlib_no-weighted_task.pkl"),
            ],
        )
        
    def _test_other_rw_methods(self):
        
        ##################################################
        #        Test-1: Read data from adj_matrix       #
        ##################################################

        # 1.1 Transfer pickle to adj_matrix
        task = MISTask()
        task.from_pickle("test_dataset/mis/task/mis_rb-small_no-weighted_task.pkl")
        adj_matrix = task.to_adj_matrix()
        
        # 1.2 Read data from adj_matrix
        new_task = MISTask()
        new_task.from_adj_matrix(adj_matrix=adj_matrix)

        # 1.3 Verify the consistency
        task_edge_index = task.edge_index
        new_task_edge_index = new_task.edge_index
        if (task_edge_index != new_task_edge_index).any():
            raise ValueError("Inconsistent data using `adj_matrix`.")

   
        ##################################################
        #    Test-2: Read data from adj_matrix_weighed   #
        ##################################################
        
        # 2.1 Transfer pickle to adj_matrix
        task = MISTask()
        task.from_pickle("test_dataset/mis/task/mis_rb-small_no-weighted_task.pkl")
        adj_matrix_weighted = task.to_adj_matrix(with_edge_weights=True)
        
        # 2.2 Read data from adj_matrix
        new_task = MISTask()
        new_task.from_adj_matrix_weighted(adj_matrix_weighted=adj_matrix_weighted)

        # 2.3 Verify the consistency
        task_edge_index = task.edge_index
        new_task_edge_index = new_task.edge_index
        if (task_edge_index != new_task_edge_index).any():
            raise ValueError("Inconsistent data using `adj_matrix_weighted`.")
        
        
        ##################################################
        #        Test-3: Read data from csr format       #
        ##################################################
    
        # 3.1 Transfer pickle to adj_matrix
        task = MISTask()
        task.from_pickle("test_dataset/mis/task/mis_rb-small_no-weighted_task.pkl")
        xadj, adjncy = task.to_csr()
        
        # 3.2 Read data from adj_matrix
        new_task = MISTask()
        new_task.from_csr(xadj=xadj, adjncy=adjncy)

        # 3.3 Verify the consistency
        task_edge_index = task.edge_index
        new_task_edge_index = new_task.edge_index
        if (task_edge_index != new_task_edge_index).any():
            raise ValueError("Inconsistent data using `csr`.")
    
    def _test_render(self):
        # Read data
        task = MISTask()
        task.from_pickle("test_dataset/mis/task/mis_rb-small_no-weighted_task.pkl")
        task.sol = task.ref_sol
        
        # Render (problem)
        tmp_path = self._make_tmp_file()
        task.render(save_path=pathlib.Path(tmp_path + ".png"), with_sol=False)
        
        # Render (solution)
        task.render(save_path=pathlib.Path(tmp_path + "_sol.png"), with_sol=True)
        
        # Clean up
        os.remove(tmp_path + ".png")
        os.remove(tmp_path + "_sol.png")