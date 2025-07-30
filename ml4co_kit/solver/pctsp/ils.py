r"""
ILS (C++) for solving PCTSPs.

Solves PCTSPs using Iterated Local Search (ILS).

We follow https://github.com/jordanamecler/PCTSP for the implementation of ILS Solver.
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


import numpy as np
from multiprocessing import Pool
from typing import Union, Tuple, List
from scipy.spatial.distance import cdist
from ml4co_kit.solver.pctsp.base import PCTSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.solver.pctsp.c_ils import pctsp_ils_solve
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class PCTSPILSSolver(PCTSPSolver):
    r"""
    Solve PCTSPs using ILS (C++).

    :param scale, int, the scale factor for coordinates.
    :param time_limit: int, the limit of running time.
    :param runs_per_instance: int, the number of runs for the ILS.
    """
    def __init__(
        self, 
        scale: int = 1e6, 
        time_limit: int = 1,
        runs_per_instance: int = 1,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(PCTSPILSSolver, self).__init__(
            solver_type=SOLVER_TYPE.ILS, scale=scale, 
            time_limit=time_limit, precision=precision
        )
        self.runs_per_instance = runs_per_instance
    
    def _solve(
        self, depots: np.ndarray, points: np.ndarray, 
        penalties: np.ndarray, norm_prizes: np.ndarray
    ) -> np.ndarray:
        r"""
        Solve a single PCTSP instance
        """            
        # Prepare the input data for the C++ solver
        coords = np.vstack([depots, points])
        dist_matrix = np.round(cdist(coords, coords, 'euclidean') * self.scale).astype(np.int32)
        prizes_full = np.round(np.insert(norm_prizes, 0, 0) * self.scale).astype(np.int32)
        penalties_full = np.round(np.insert(penalties, 0, 0) * self.scale).astype(np.int32)
        min_prize_scaled = int(1.0 * self.scale)
               
        # Call the C++ solver with the scaled min_prize
        tour = pctsp_ils_solve(
            dist_matrix=dist_matrix,
            penalties=penalties_full,
            prizes=prizes_full,
            min_prize_scaled=min_prize_scaled,
            runs_per_instance=self.runs_per_instance 
        )

        return tour
    
    def solve(
        self,
        depots: Union[np.ndarray, List[np.ndarray]] = None,
        points: Union[np.ndarray, List[np.ndarray]] = None,
        penalties: Union[np.ndarray, List[np.ndarray]] = None,
        norm_prizes: Union[np.ndarray, List[np.ndarray]] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (costs, routes)
        r"""
        Solve PCTSP instances using ILS (C++).
        """        
        # preparation
        self.from_data(
            depots=depots, 
            points=points, 
            penalties=penalties, 
            norm_prizes=norm_prizes, 
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
                    self.depots[idx], self.points[idx], 
                    self.penalties[idx], self.norm_prizes[idx]
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
                             self.norm_prizes[idx*num_threads+inner_idx])
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
        return "PCTSPILSSolver"