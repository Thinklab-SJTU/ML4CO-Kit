r"""
This module provides a class ATSPLKHSolver for solving the ATSP
using the LKH (Lin-Kernighan Heuristic) algorithm.
LKH is a heuristic algorithm that uses k-opt move strategies
to find approximate optimal solutions to problems.
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
import uuid
import pathlib
import numpy as np
from typing import Union
from multiprocessing import Pool
from subprocess import check_call
from ml4co_kit.solver.atsp.base import ATSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class ATSPLKHSolver(ATSPSolver):
    r"""
    Solve ATSPs using LKH solver.

    :param lkh_max_trials, int, the maximum number of trials for the LKH solver.
    :param lkh_path, pathlib.Path, the path of the LKH solver.
    :param scale, int, the scale factor for coordinates.
    :param lkh_runs, int, the number of runs for the LKH solver.
    :param lkh_seed, int, the random number seed for the LKH solver.
    :param lkh_special, boolean, whether to solve in a special way.
    """
    def __init__(
        self,
        scale: int = 1e6,
        lkh_max_trials: int = 500,
        lkh_path: pathlib.Path = "LKH",
        lkh_runs: int = 1,
        lkh_seed: int = 1234,
        lkh_special: bool = False
    ):
        super(ATSPLKHSolver, self).__init__(solver_type=SOLVER_TYPE.LKH, scale=scale)
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_runs = lkh_runs
        self.lkh_seed = lkh_seed
        self.lkh_special = lkh_special
        
    def _write_parameter_file(
        self,
        save_path: str,
        atsp_file_path: str,
        tour_path: str
    ):
        r"""
        Writing LKH solving parameters."""
        with open(save_path, "w") as f:
            f.write(f"PROBLEM_FILE = {atsp_file_path}\n")
            f.write(f"MAX_TRIALS = {self.lkh_max_trials}\n")
            f.write(f"RUNS = {self.lkh_runs}\n")
            f.write(f"SEED = {self.lkh_seed}\n")
            f.write(f"TOUR_FILE = {tour_path}\n")
            if self.lkh_special:
                f.write("SPECIAL\n")
    
    def _solve(self, dist: np.ndarray) -> np.ndarray:
        r"""
        Solve a single ATSP instance.
        """
        # Intermediate files
        tmp_name = uuid.uuid4().hex[:9]
        para_save_path = f"{tmp_name}.para"
        atsp_save_path = f"{tmp_name}.atsp"
        tour_save_path = f"{tmp_name}.opt.tour"
        log_save_path = f"{tmp_name}.log"
        
        # prepare for solve
        self.tmp_solver.from_data(dist * self.tmp_solver.scale)
        self.tmp_solver.to_tsplib_folder(
            atsp_save_dir="./", 
            atsp_filename=atsp_save_path
        )
        self._write_parameter_file(
            save_path=para_save_path,
            atsp_file_path=atsp_save_path,
            tour_path=tour_save_path
        )
        
        # solve
        with open(log_save_path, "w") as f:
            check_call([self.lkh_path, para_save_path], stdout=f)
            
        # read solution
        self.tmp_solver.from_tsplib(tour_file_path=tour_save_path)
        tour = self.tmp_solver.tours[0]
        
        # delete files
        files_path = [
            para_save_path, atsp_save_path, tour_save_path, log_save_path
        ]
        for file_path in files_path:
           if os.path.exists(file_path):
               os.remove(file_path)
        
        # return
        return tour

    def solve(
        self,
        dists: Union[np.ndarray, list] = None,
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        r"""
        Solve ATSP using LKH algorithm with options for normalization,
        threading, and timing.

        :param dists: np.ndarry or list, the dists matrix to solve TSP for. Defaults to None.
        :param normalize: boolean, whether to normalize the points. Defaults to "False".
        :param num_threads: int, The number of threads to use for solving. Defaults to 1.
        :param show_time: boolean, whether to show the time taken to solve. Defaults to "False".

        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import ATSPLKHSolver
                
                # create ATSPLKHSolver
                >>> solver = ATSPLKHSolver(lkh_max_trials=1)

                # load data and reference solutions from ``.tsp`` file
                >>> solver.from_tsplib(
                        atsp_file_path="examples/atsp/tsplib_1/problem/ft53.atsp",
                        ref=False,
                        normalize=True
                    )
                    
                # solve
                >>> solver.solve()
                [[ 0,  5,  9, 11,  3,  8,  4,  1,  2, 10,  7,  6,  0]]
         """
    
        # prepare
        self.from_data(dists=dists, normalize=normalize)
        self.tmp_solver = ATSPSolver(scale=self.scale)
        timer = Timer(apply=show_time)
        timer.start()

        # solve
        tours = list()
        dists_shape = self.dists.shape
        num_dists = dists_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_dists, self.solve_msg, show_time):
                tours.append(self._solve(self.dists[idx]))
        else:
            batch_dists = self.dists.reshape(-1, num_threads, dists_shape[-2], dists_shape[-1])
            for idx in iterative_execution(
                range, num_dists // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.map(
                        self._solve,
                        [
                            batch_dists[idx][inner_idx]
                            for inner_idx in range(num_threads)
                        ],
                    )
                for tour in cur_tours:
                    tours.append(tour)
        
        # format
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        return self.tours
    
    def __str__(self) -> str:
        return "ATSPLKHSolver"