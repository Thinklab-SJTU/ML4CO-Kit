r"""
TSP Task Tester.
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
from ml4co_kit import TSPTask
from tests.task_test.base import TaskTesterBase


class TSPTaskTester(TaskTesterBase):
    def __init__(self):
        super(TSPTaskTester, self).__init__(
            test_task_class=TSPTask,
            pickle_files_list=[
                pathlib.Path("test_dataset/tsp/task/tsp50_cluster_task.pkl"),
                pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_task.pkl"),
                pathlib.Path("test_dataset/tsp/task/tsp50_uniform_task.pkl"),
                pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl"),
            ],
        )
        
    def _test_other_rw_methods(self):

        ##################################################
        #       Test-1: Read data from TSPLIB file       #
        ##################################################
        
        # 1.1 Read data from TSPLIB file
        task = TSPTask()
        task.from_tsplib(
            tsp_file_path=pathlib.Path("test_dataset/tsp/tsplib/problem/a280.tsp"),
            tour_file_path=pathlib.Path("test_dataset/tsp/tsplib/solution/a280.opt.tour"),
        )
        
        # 1.2 Evaluate the solution
        cost = task.evaluate(task.sol)
        print(f"TSP-50 Cluster: {cost}")
        
        # 1.3 Output data using ``to_tsplib``
        tmp_name = self._make_tmp_file()
        tmp_tsp_file_path = pathlib.Path(tmp_name + "_tsp")
        tmp_tour_file_path = pathlib.Path(tmp_name + "_tour")
        task.to_tsplib(
            tsp_file_path=tmp_tsp_file_path,
            tour_file_path=tmp_tour_file_path,
        )
        
        # 1.4 Verify the consistency
        new_task = TSPTask()
        new_task.from_tsplib(
            tsp_file_path=tmp_tsp_file_path,
            tour_file_path=tmp_tour_file_path,
        )
        new_cost = new_task.evaluate(new_task.sol)
        print(f"TSP-50 Cluster: {new_cost}")
        if cost != new_cost:
            raise ValueError("Inconsistent TSPLIB data.")
        
        # 1.5 Clean up
        os.remove(tmp_tsp_file_path)
        os.remove(tmp_tour_file_path)
        
    def _test_render(self):
        # Read data
        task = TSPTask()
        task.from_pickle("test_dataset/tsp/task/tsp50_cluster_task.pkl")
        task.sol = task.ref_sol
        
        # Render (problem)
        tmp_path = self._make_tmp_file()
        task.render(save_path=pathlib.Path(tmp_path + ".png"), with_sol=False)
        
        # Render (solution)
        task.render(save_path=pathlib.Path(tmp_path + "_sol.png"), with_sol=True)
        
        # Clean up
        os.remove(tmp_path + ".png")
        os.remove(tmp_path + "_sol.png")