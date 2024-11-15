import os
import uuid
import numpy as np
from typing import Union
from multiprocessing import Pool
from ml4co_kit.solver.tsp.base import TSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.solver.tsp.pyconcorde import TSPConSolver
from ml4co_kit.utils.time_utils import iterative_execution, Timer


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
        super(TSPConcordeSolver, self).__init__(
            solver_type=SOLVER_TYPE.CONCORDE, scale=scale
        )

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
        # preparation
        self.from_data(points=points, norm=norm, normalize=normalize)
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        tours = list()
        p_shape = self.points.shape
        num_points = p_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                name = uuid.uuid4().hex
                tours.append(self._solve(self.points[idx], name))
                self.clear_tmp_files(name)
        else:
            batch_points = self.points.reshape(-1, num_threads, p_shape[-2], p_shape[-1])
            name_list = list()
            for idx in iterative_execution(
                range, num_points // num_threads, self.solve_msg, show_time
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

        # format
        tours = np.array(tours)
        zeros = np.zeros((tours.shape[0], 1))
        tours = np.append(tours, zeros, axis=1).astype(np.int32)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return self.tours

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

    def __str__(self) -> str:
        return "TSPConcordeSolver"