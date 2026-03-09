r"""
Test File Utils Module.
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
import shutil
from ml4co_kit import (
    download, pull_file_from_huggingface, get_md5, 
    extract_archive, compress_folder, points_augment, 
    graph_augment, TSPTask, MISTask, ConcordeSolver
)


class AugmentUtilsTester(object):
    """Test cases for augmentation utility functions."""
    
    def __init__(self) -> None:
        pass
    
    def test(self):
        
        ###############################################
        #          Test-1 Augment the points          #
        ###############################################
        
        # 1.1 get tsp task
        tsp_task = TSPTask()
        tsp_task.from_pickle(file_path="test_dataset/routing/tsp/task/tsp50_uniform_task.pkl")
        
        # 1.2 augment the points
        augmented_points = points_augment(tsp_task.points.copy())
        
        # 1.3 check the consistency for sol
        solver = ConcordeSolver()
        tsp_task.from_data(points=augmented_points)
        solver.solve(tsp_task)
        gap = tsp_task.evaluate_w_gap()[2]
        if abs(gap) > 1e-5:
            raise ValueError("Inconsistent gap for the augmented points.")

        ###############################################
        #       Test-2 Augment the graph and sol      #
        ###############################################
        
        # 2.1 get mis task
        mis_task = MISTask()
        mis_task.from_pickle(file_path="test_dataset/graph/mis/task/mis_rb-small_no-weighted_task.pkl")
        
        # 2.2 augment the graph and sol
        augmented_graph, _, augmented_ref_sol = \
            graph_augment(
                graph=mis_task.to_adj_matrix().copy(), 
                ref_sol=mis_task.ref_sol.copy()
            )

        # 2.3 check the correctness of the augmented graph and sol
        mis_task.from_adj_matrix(adj_matrix=augmented_graph)
        mis_task.check_constraints(sol=augmented_ref_sol)