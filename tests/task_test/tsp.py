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
import numpy as np
from ml4co_kit import TSPTask, DISTANCE_TYPE
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
        
        
        ##################################################
        #       Test-2: Validate various distances       #
        ##################################################
        
        # Basic
        sol = np.arange(20)
        sol = np.append(sol, 0)
        
        # 2.1 EUC_2D
        task_euc_2d = TSPTask(distance_type=DISTANCE_TYPE.EUC_2D)
        points_2d = np.random.uniform(0, 1, size=(20, 2))
        task_euc_2d.from_data(points=points_2d)
        cost_euc_2d = task_euc_2d.evaluate(sol)
        print(f"EUC_2D cost: {cost_euc_2d}")

        # 2.2 EUC_3D
        task_euc_3d = TSPTask(distance_type=DISTANCE_TYPE.EUC_3D)
        points_3d = np.random.uniform(0, 1, size=(20, 3))
        task_euc_3d.from_data(points=points_3d)
        cost_euc_3d = task_euc_3d.evaluate(sol)
        print(f"EUC_3D cost: {cost_euc_3d}")
        
        # 2.3 MAX_2D
        task_max_2d = TSPTask(distance_type=DISTANCE_TYPE.MAX_2D)
        task_max_2d.from_data(points=points_2d)
        cost_max_2d = task_max_2d.evaluate(sol)
        print(f"MAX_2D cost: {cost_max_2d}")
        
        # 2.4 MAX_3D
        task_max_3d = TSPTask(distance_type=DISTANCE_TYPE.MAX_3D)
        task_max_3d.from_data(points=points_3d)
        cost_max_3d = task_max_3d.evaluate(sol)
        print(f"MAX_3D cost: {cost_max_3d}")
        
        # 2.5 MAN_2D
        task_man_2d = TSPTask(distance_type=DISTANCE_TYPE.MAN_2D)
        task_man_2d.from_data(points=points_2d)
        cost_man_2d = task_man_2d.evaluate(sol)
        print(f"MAN_2D cost: {cost_man_2d}")
        
        # 2.6 MAN_3D
        task_man_3d = TSPTask(distance_type=DISTANCE_TYPE.MAN_3D)
        task_man_3d.from_data(points=points_3d)
        cost_man_3d = task_man_3d.evaluate(sol)
        print(f"MAN_3D cost: {cost_man_3d}")
        
        # 2.7 GEO
        task_geo = TSPTask(distance_type=DISTANCE_TYPE.GEO)
        # For GEO distance, we need geographical coordinates (latitude, longitude)
        # Generate random coordinates in a reasonable geographical range
        lat_coords = np.random.uniform(-90, 90, size=(20, 1))  # Latitude: -90 to 90
        lon_coords = np.random.uniform(-180, 180, size=(20, 1))  # Longitude: -180 to 180
        points_geo = np.hstack([lat_coords, lon_coords])
        task_geo.from_data(points=points_geo)
        cost_geo = task_geo.evaluate(sol)
        print(f"GEO cost: {cost_geo}")
        
        # 2.8 ATT
        task_att = TSPTask(distance_type=DISTANCE_TYPE.ATT)
        task_att.from_data(points=points_2d)
        cost_att = task_att.evaluate(sol)
        print(f"ATT cost: {cost_att}")

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