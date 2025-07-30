r"""
REOPT for solving SPCTSPs.

The algorithm plans a tour, executes part of it and then re-optimizes using the C++ ILS algorithm.
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
import subprocess
import numpy as np
from multiprocessing import Pool
from typing import Union, Tuple, List
from scipy.spatial.distance import cdist
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.solver.spctsp.base import SPCTSPSolver
from ml4co_kit.solver.spctsp.c_reopt import spctsp_reopt_solve
from ml4co_kit.utils.time_utils import iterative_execution, Timer

class SPCTSPReoptSolver(SPCTSPSolver):
    r"""
    Solve SPCTSPs using REOPT.

    :param scale, int, the scale factor for coordinates.
    :param time_limit: int, the limit of running time.
    :param cpp_solver_path: str, the path to the C++ solver executable.
    :param runs_per_instance: int, the number of runs for the ILS.
    :param append: str, the method to append the tour.
    """
    def __init__(
        self, 
        scale: int = 1e7, 
        time_limit: int = 1,
        runs_per_instance: int = 1,
        append_strategy: str = "half"
    ):
        super(SPCTSPReoptSolver, self).__init__(
            solver_type=SOLVER_TYPE.REOPT, scale=scale
        )
        self.time_limit = time_limit
        self.runs_per_instance = runs_per_instance
        self.append_strategy = append_strategy
    
    def _solve(
        self, depots: np.ndarray, points: np.ndarray, penalties: np.ndarray, 
        norm_prizes: np.ndarray, stochastic_norm_prizes: np.ndarray
    ) -> np.ndarray:
        r"""
        Solve a single PCTSP instance
        """       
        # Prepare the input data for the C++ solver
        coords = np.vstack([depots, points])
        dist_matrix = cdist(coords, coords, 'euclidean')
        
        # Call the C++ solver
        tour = spctsp_reopt_solve(
            dist_matrix=dist_matrix, 
            penalties=penalties, 
            norm_prizes=norm_prizes, 
            stochastic_norm_prizes=stochastic_norm_prizes,
            runs_per_instance=self.runs_per_instance,
            scale=self.scale,
            append_strategy=self.append_strategy
        )
        final_tour = [0] + tour + [0]
        return final_tour

    def solve(
        self,
        depots: Union[np.ndarray, List[np.ndarray]] = None,
        points: Union[np.ndarray, List[np.ndarray]] = None,
        penalties: Union[np.ndarray, List[np.ndarray]] = None,
        norm_prizes: Union[np.ndarray, List[np.ndarray]] = None,
        stochastic_norm_prizes: Union[np.ndarray, List[np.ndarray]] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (costs, routes)
        r"""
        Solve SPCTSP instances using REOPT.
        """        
        # preparation
        self.from_data(
            depots=depots, 
            points=points, 
            penalties=penalties, 
            norm_prizes=norm_prizes, 
            stochastic_norm_prizes=stochastic_norm_prizes, 
            norm=norm, 
            normalize=normalize
        )
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                tours.append(self._solve(
                    self.depots[idx], self.points[idx], self.penalties[idx], 
                    self.norm_prizes[idx], self.stochastic_norm_prizes[idx]
                ))
        else:
            for idx in iterative_execution(
                range, num_points // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.starmap(
                        self._solve,
                        [
                            (self.depots[idx*num_threads+inner_idx],
                             self.points[idx*num_threads+inner_idx],
                             self.penalties[idx*num_threads+inner_idx],
                             self.norm_prizes[idx*num_threads+inner_idx],
                             self.stochastic_norm_prizes[idx*num_threads+inner_idx])
                            for inner_idx in range(num_threads)
                        ],
                    )
                for tour in cur_tours:
                    tours.append(tour)
        
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return self.tours

    def __str__(self) -> str:
        return "SPCTSPReoptSolver"