import os
import time
import uuid
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from typing import Union
from .pyconcorde import TSPConSolver
from .base import TSPSolver


class TSPConcordeSolver(TSPSolver):
    def __init__(
        self,
        scale: int = 1e6,
    ):
        """
        TSPConcordeSolver
        Args:
            scale (int, optional):
                The scale factor for coordinates in the Concorde solver.
        """
        super(TSPConcordeSolver, self).__init__(solver_type="concorde", scale=scale)

    def _solve(self, nodes_coord: np.ndarray, name: str) -> np.ndarray:
        solver = TSPConSolver.from_data(
            xs=nodes_coord[:, 0] * self.scale,
            ys=nodes_coord[:, 1] * self.scale,
            norm=self.norm,
            name=name,
        )
        solution = solver.solve(verbose=False, name=name)
        tour = solution.tour
        return tour

    def solve(
        self,
        points: Union[np.ndarray, list] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> np.ndarray:
        # prepare
        self.from_data(points, norm, normalize)
        start_time = time.time()

        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            if show_time:
                for idx in tqdm(range(num_points), desc="Solving TSP Using Concorde"):
                    name = uuid.uuid4().hex
                    tours.append(self._solve(self.points[idx], name))
                    self.clear_tmp_files(name)
            else:
                for idx in range(num_points):
                    name = uuid.uuid4().hex
                    tours.append(self._solve(self.points[idx], name))
                    self.clear_tmp_files(name)
        else:
            batch_points = self.points.reshape(
                -1, num_threads, p_shape[-2], p_shape[-1]
            )
            name_list = list()
            if show_time:
                for idx in tqdm(
                    range(num_points // num_threads), desc="Solving TSP Using Concorde"
                ):
                    for _ in range(num_threads):
                        name_list.append(uuid.uuid4().hex)
                    with Pool(num_threads) as p1:
                        name = uuid.uuid4().hex
                        cur_tours = p1.starmap(
                            self._solve,
                            [
                                (batch_points[idx][inner_idx], name)
                                for inner_idx, name in zip(
                                    range(num_threads), name_list
                                )
                            ],
                        )
                    for tour in cur_tours:
                        tours.append(tour)
                    for name in name_list:
                        self.clear_tmp_files(name)
            else:
                for idx in range(num_points // num_threads):
                    for _ in range(num_threads):
                        name_list.append(uuid.uuid4().hex)
                    with Pool(num_threads) as p1:
                        name = uuid.uuid4().hex
                        cur_tours = p1.starmap(
                            self._solve,
                            [
                                (batch_points[idx][inner_idx], name)
                                for inner_idx, name in zip(
                                    range(num_threads), name_list
                                )
                            ],
                        )
                    for tour in cur_tours:
                        tours.append(tour)
                    for name in name_list:
                        self.clear_tmp_files(name)

        # format
        tours = np.array(tours)
        zeros = np.zeros((tours.shape[0], 1))
        tours = np.append(tours, zeros, axis=1).astype(np.int32)
        if tours.ndim == 2 and tours.shape[0] == 1:
            tours = tours[0]
        self.read_tours(tours)
        end_time = time.time()
        if show_time:
            print(f"Use Time: {end_time - start_time}")
        return tours

    def clear_tmp_files(self, name):
        real_name = name[0:9]
        # tmp file
        sol_filename = f"{real_name}.sol"
        Osol_filename = f"O{real_name}.sol"
        res_filename = f"{real_name}.res"
        Ores_filename = f"O{real_name}.res"
        sav_filename = f"{real_name}.sav"
        Osav_filename = f"O{real_name}.sav"
        pul_filename = f"{real_name}.pul"
        Opul_filename = f"O{real_name}.pul"
        filelist = [
            sol_filename,
            Osol_filename,
            res_filename,
            Ores_filename,
            sav_filename,
            Osav_filename,
            pul_filename,
            Opul_filename,
        ]
        # intermediate file
        for i in range(100):
            filelist.append("{}.{:03d}".format(name[0:8], i + 1))
        # delete
        for file in filelist:
            if os.path.exists(file):
                os.remove(file)
