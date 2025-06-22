r"""
Basic solver for Prize Collection Traveling Salesman Problem (PCTSP). 

In the PCTSP, each node has an associated prize and penalty. Unlike the traditional TSP, where 
all cities must be visited, the PCTSP allows for skipping some nodes. The final objective is 
to minimize the sum of the traveled distance and the sum of the penalties for unvisited nodes 
while satisfying a minimum total prize constraint.
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
import sys
import numpy as np
from typing import Union
from ml4co_kit.utils import tsplib95
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.evaluate.tsp.base import TSPEvaluator
from ml4co_kit.utils.type_utils import to_numpy, TASK_TYPE, SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, iterative_execution_for_file


SUPPORT_NORM_TYPE = ["EUC_2D", "GEO"]
if sys.version_info.major == 3 and sys.version_info.minor == 8:
    from pyvrp.read import ROUND_FUNCS
else:
    from ml4co_kit.utils.round import ROUND_FUNCS


class PCTSPSolver(SolverBase):
    r"""
    This class provides a basic framework for solving PCTSP problems. It includes methods for 
    loading and outputting data in various file formats, normalizing points, and evaluating 
    solutions. Note that the actual solving method should be implemented in subclasses.
    
    :param nodes_num: :math:`N`, int, the number of nodes in TSP problem.
    :param depots: :math:`(B\times 2)`, np.ndarray, the coordinates of depots.
    :param ori_points: :math:`(B\times N \times 2)`, np.ndarray, the original coordinates data read.
    :param points: :math:`(B\times N \times 2)`, np.ndarray, the coordinates data called 
        by the solver during solving. They may initially be the same as ``ori_points``,
        but may later undergo standardization or scaling processing.
    :param penalties: :math:`(B\times N)`, np.ndarray, the penalties of nodes.
    :param deterministic_prizes: :math:`(B\times N)`, np.ndarray, the deterministic prizes of nodes.
    :param stochastic_prizes: :math:`(B\times N)`, np.ndarray, the stochastic prizes of nodes.
    :param tours: :math:`(B\times (N+1))`, np.ndarray, the solutions to the problems. 
    :param ref_tours: :math:`(B\times (N+1))`, np.ndarray, the reference solutions to the problems. 
    :param scale: int, magnification scale of coordinates. If the input coordinates are too large,
        you can scale them to 0-1 by setting ``normalize`` to True, and then use ``scale`` to adjust them.
        Note that the magnification scale only applies to ``points`` when solved by the solver.
    :param norm: string, coordinate type. It can be a 2D Euler distance or geographic data type.
    """
    def __init__(
        self, 
        solver_type: SOLVER_TYPE = None, 
        scale: int = 1e6,
        time_limit: float = 60.0
    ):
        super(PCTSPSolver, self).__init__(
            task_type=TASK_TYPE.TSP, solver_type=solver_type
        )
        self.scale: np.ndarray = scale
        self.time_limit: float = time_limit
        self.nodes_num: int = None
        self.depots: np.ndarray = None
        self.points: np.ndarray = None
        self.ori_points: np.ndarray = None
        self.penalties: np.ndarray = None
        self.deterministic_prizes: np.ndarray = None
        self.stochastic_prizes: np.ndarray = None
        self.tours: np.ndarray = None
        self.ref_tours: np.ndarray = None
        self.norm: str = None
        
    def _check_depots_dim(self):
        r"""
        Ensures that the ``depots`` attribute is a 2D array. If ``depots`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``depots``
        is neither 1D nor 2D.
        """
        if self.depots is not None:
            if self.depots.ndim == 1:
                self.depots = np.expand_dims(self.depots, axis=0)
            if self.depots.ndim != 2:
                raise ValueError("The dimensions of ``depots`` cannot be larger than 2.")
        
    def _check_points_dim(self):
        r"""
        Ensures that the ``points`` attribute is a 3D array. If ``points`` is a 2D array,
        it adds an additional dimension to make it 3D. Raises a ``ValueError`` if ``points``
        is neither 2D nor 3D. Also sets the ``nodes_num`` attribute to the number of nodes
        (points) in the problem.
        """
        if self.points is not None:
            if self.points.ndim == 2:
                self.points = np.expand_dims(self.points, axis=0)
            if self.points.ndim != 3:
                raise ValueError("``points`` must be a 2D or 3D array.")
            self.nodes_num = self.points.shape[1]

    def _check_ori_points_dim(self):
        r"""
        Ensures that the ``ori_points`` attribute is a 3D array. Calls ``_check_points_dim``
        to validate the ``points`` attribute first. If ``ori_points`` is a 2D array, it adds
        an additional dimension to make it 3D. Raises a ``ValueError`` if ``ori_points`` is
        neither 2D nor 3D.
        """
        self._check_points_dim()
        if self.ori_points is not None:
            if self.ori_points.ndim == 2:
                self.ori_points = np.expand_dims(self.ori_points, axis=0)
            if self.ori_points.ndim != 3:
                raise ValueError("The ``ori_points`` must be 2D or 3D array.")
            
    def _check_prizes_dim(self):
        r"""
        Ensures that the ``deterministic_prizes`` and ``stochastic_prizes`` attributes
        are 2D arrays. If either is a 1D array, it adds an additional dimension
        to make it 2D. Raises a ``ValueError`` if either has more than 2 dimensions.
        """
        if self.deterministic_prizes is not None:
            if self.deterministic_prizes.ndim == 1:
                self.deterministic_prizes = np.expand_dims(self.deterministic_prizes, axis=0)
            if self.deterministic_prizes.ndim != 2:
                raise ValueError("The dimensions of ``deterministic_prizes`` cannot be larger than 2.")
        if self.stochastic_prizes is not None:
            if self.stochastic_prizes.ndim == 1:
                self.stochastic_prizes = np.expand_dims(self.stochastic_prizes, axis=0)
            if self.stochastic_prizes.ndim != 2:
                raise ValueError("The dimensions of ``stochastic_prizes`` cannot be larger than 2.")

    def _check_tours_dim(self):
        r"""
        Ensures that the ``tours`` attribute is a 2D array. If ``tours`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``tours``
        has more than 2 dimensions.
        """
        if self.tours is not None:
            if self.tours.ndim == 1:
                self.tours = np.expand_dims(self.tours, axis=0)
            if self.tours.ndim != 2:
                raise ValueError("The dimensions of ``tours`` cannot be larger than 2.")

    def _check_ref_tours_dim(self):
        r"""
        Ensures that the ``ref_tours`` attribute is a 2D array. If ``ref_tours`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``ref_tours``
        has more than 2 dimensions.
        """
        if self.ref_tours is not None:
            if self.ref_tours.ndim == 1:
                self.ref_tours = np.expand_dims(self.ref_tours, axis=0)
            if self.ref_tours.ndim != 2:
                raise ValueError(
                    "The dimensions of the ``ref_tours`` cannot be larger than 2."
                )
        
    def _check_depots_not_none(self):
        r"""
        Checks if the ``depots`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``depots`` is ``None``. 
        """
        if self.depots is None:
            message = (
                "``depots`` cannot be None! You can load the ``depots`` using the methods including "
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)
        
    def _check_points_not_none(self):
        r"""
        Checks if the ``points`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``points`` is ``None``. 
        """
        if self.points is None:
            message = (
                "``points`` cannot be None! You can load the ``points`` using the methods including "
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)
        
    def _check_penalties_not_none(self):
        r"""
        Checks if the ``penalties`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``penalties`` is ``None``. 
        """
        if self.penalties is None:
            message = (
                "``penalties`` cannot be None! You can load the ``penalties`` using the methods including "
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)
        
    def _check_prizes_not_none(self):
        r"""
        Checks if the ``deterministic_prizes`` or ``stochastic_prizes`` attributes are not ``None``.
        Raises a ``ValueError`` if both attributes are ``None``.
        """
        if self.deterministic_prizes is None or self.stochastic_prizes is None:
            message = (
                "Both ``deterministic_prizes`` and ``stochastic_prizes`` cannot be None! "
                "You can load the prizes using the methods including "
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)

    def _check_tours_not_none(self, ref: bool):
        r"""
        Checks if the ``tours` or ``ref_tours`` attribute is not ``None``.
        - If ``ref`` is ``True``, it checks the ``ref_tours`` attribute.
        - If ``ref`` is ``False``, it checks the ``tours`` attribute.
        Raises a `ValueError` if the respective attribute is ``None``.
        """
        msg = "ref_tours" if ref else "tours"
        message = (
            f"``{msg}`` cannot be None! You can use solvers based on ``OPSolver``"
            "or use methods including ``from_data`` or  ``from_txt`` to obtain them."
        )  
        if ref:
            if self.ref_tours is None:
                raise ValueError(message)
        else:
            if self.tours is None:    
                raise ValueError(message)

    def _set_norm(self, norm: str):
        r"""
        Sets the coordinate type.
        """
        if norm is None:
            return
        if norm not in SUPPORT_NORM_TYPE:
            message = (
                f"The norm type ({norm}) is not a valid type, "
                f"only {SUPPORT_NORM_TYPE} are supported."
            )
            raise ValueError(message)
        if norm == "GEO" and self.scale != 1:
            message = "The scale must be 1 for ``GEO`` norm type."
            raise ValueError(message)
        self.norm = norm

    def _normalize_points(self):
        r"""
        Normalizes the ``points`` attribute to scale all coordinates between 0 and 1.
        """
        for idx in range(self.points.shape[0]):
            cur_points = self.points[idx]
            max_value = np.max(cur_points)
            min_value = np.min(cur_points)
            cur_points = (cur_points - min_value) / (max_value - min_value)
            self.points[idx] = cur_points

    def _get_round_func(self, round_func: str):
        r"""
        Retrieves a rounding function based on the input string or function.
        - If `round_func` is a string, it checks against predefined functions (``ROUND_FUNCS``).
        - If `round_func` is not callable, raises a ``TypeError``.
        """
        if (key := str(round_func)) in ROUND_FUNCS:
            round_func = ROUND_FUNCS[key]
        if not callable(round_func):
            raise TypeError(
                f"round_func = {round_func} is not understood. Can be a function,"
                f" or one of {ROUND_FUNCS.keys()}."
            )
        return round_func

    def _apply_scale_and_dtype(
        self, points: np.ndarray, apply_scale: bool, to_int: bool, round_func: str
    ):
        r"""
        Applies scaling and/or dtype conversion to the given ``points``.
        - Scales the points by ``self.scale`` if ``apply_scale`` is True.
        - Converts points to integers using the specified rounding function if ``to_int`` is True.
        """
        # apply scale
        if apply_scale:
            points = points * self.scale

        # dtype
        if to_int:
            round_func = self._get_round_func(round_func)
            points = round_func(points)
        
        return points        
        
    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time: bool = False
    ):
        """
        Read data from `.txt` file.

        :param file_path: string, path to the `.txt` file containing TSP instances data.
        :param ref: boolean, whether the solution is a reference solution.
        :param return_list: boolean, only use this function to obtain data, but do not save it to the solver. 
        :param norm: boolean, the normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import TSPSolver
                
                # create TSPSolver
                >>> solver = TSPSolver()

                # load data from ``.txt`` file
                >>> solver.from_txt(file_path="examples/tsp/txt/tsp50_concorde.txt")
                >>> solver.points.shape
                (16, 50, 2)
                >>> solver.tours.shape
                (16, 51)
        """
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            points_list = list()
            tour_list = list()
            load_msg = f"Loading data from {file_path}"
            for line in iterative_execution_for_file(file, load_msg, show_time):
                line = line.strip()
                split_line = line.split(" output ")
                points = split_line[0]
                tour = split_line[1]
                tour = tour.split(" ")
                tour = np.array([int(t) for t in tour])
                tour -= 1
                tour_list.append(tour)
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ]
                )
                points_list.append(points)

        # check if return list
        if return_list:
            return points_list, tour_list

        # check tours
        try:
            points = np.array(points_list)
            tours = np.array(tour_list)
        except Exception as e:
            message = (
                "This method does not support instances of different numbers of nodes. "
                "If you want to read the data, please set ``return_list`` as True. "
                "Anyway, the data will not be saved in the solver. "
                "Please convert the data to ``np.ndarray`` externally before calling the solver."
            )
            raise Exception(message) from e

        # use ``from_data``
        self.from_data(
            points=points, tours=tours, ref=ref, norm=norm, normalize=normalize
        )

    def from_data(
        self,
        points: Union[list, np.ndarray] = None,
        tours: Union[list, np.ndarray] = None,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
    ):
        """
        Read data from list or np.ndarray.

        :param points: np.ndarray, the coordinates of nodes. If given, the points 
            originally stored in the solver will be replaced.
        :param tours: np.ndarray, the solutions of the problems. If given, the tours
            originally stored in the solver will be replaced
        :param ref: boolean, whether the solution is a reference solution.
        :param norm: string, the normalization type for node coordinates (default is "EUC_2D").
        :param normalize: boolean, Whether to normalize node coordinates.

        .. dropdown:: Example

            :: 

                >>> import numpy as np
                >>> from ml4co_kit import TSPSolver
                
                # create TSPSolver
                >>> solver = TSPSolver()

                # load data from np.ndarray
                >>> solver.from_data(points=np.random.random(size=(10, 2)))
                >>> solver.points.shape
                (1, 10, 2)
        """
        # set norm
        self._set_norm(norm)
    
        # points
        if points is not None:
            points = to_numpy(points)
            self.ori_points = points
            self.points = points.astype(np.float32)
            self._check_ori_points_dim()
            if normalize:
                self._normalize_points()
    
        # tours
        if tours is not None:
            tours = to_numpy(tours).astype(np.int32)
            if ref:
                self.ref_tours = tours
                self._check_ref_tours_dim()
            else:
                self.tours = tours
                self._check_tours_dim()

    def to_txt(
        self,
        file_path: str = "example.txt",
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round"
    ):
        """
        Output(store) data in ``txt`` format

        :param file_path: string, path to save the `.txt` file.
        :param tour_file_path: string, path to the `.tour` file containing TSP solution data.
            if given, the solver will read tour from the file.
        :param original: boolean, whether to use ``original points`` or ``points``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.

        .. note::
            ``points`` and ``tours`` must not be None.
         
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import TSPSolver
                
                # create TSPSolver
                >>> solver = TSPSolver()

                # load data from ``.tsp`` and ``.opt.tour`` files
                >>> solver.from_tsplib(
                        tsp_file_path="examples/tsp/tsplib_1/problem/kroC100.tsp",
                        tour_file_path="examples/tsp/tsplib_1/solution/kroC100.opt.tour",
                        ref=False,
                        norm="EUC_2D",
                        normalize=True
                    )
                    
                # Output data in ``txt`` format
                >>> solver.to_txt("kroC100.txt")
        """
        # check
        self._check_points_not_none()
        self._check_tours_not_none(ref=False)
        
        # variables
        points = self.ori_points if original else self.points
        tours = self.tours

        # deal with different shapes
        samples = points.shape[0]
        if tours.shape[0] != samples:
            # a problem has more than one solved tour
            samples_tours = tours.reshape(samples, -1, tours.shape[-1])
            best_tour_list = list()
            for idx, solved_tours in enumerate(samples_tours):
                cur_eva = TSPEvaluator(points[idx])
                best_tour = solved_tours[0]
                best_cost = cur_eva.evaluate(best_tour)
                for tour in solved_tours:
                    cur_cost = cur_eva.evaluate(tour)
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_tour = tour
                best_tour_list.append(best_tour)
            tours = np.array(best_tour_list)

        # apply scale and dtype
        points = self._apply_scale_and_dtype(
            points=points, apply_scale=apply_scale,
            to_int=to_int, round_func=round_func
        )

        # write
        with open(file_path, "w") as f:
            for node_coordes, tour in zip(points, tours):
                f.write(" ".join(str(x) + str(" ") + str(y) for x, y in node_coordes))
                f.write(str(" ") + str("output") + str(" "))
                f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                f.write("\n")
            f.close()

    def evaluate(
        self,
        calculate_gap: bool = False,
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round",
    ):
        """
        Evaluate the solution quality of the solver

        :param calculate_gap: boolean, whether to calculate the gap with the reference solutions.
        :param original: boolean, whether to use ``original points`` or ``points``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.

        .. note::
            - Please make sure the ``points`` and the ``tours`` are not None.
            - If you set the ``calculate_gap`` as True, please make sure the ``ref_tours`` is not None.
        
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import TSPLKHSolver
                
                # create TSPLKHSolver
                >>> solver = TSPLKHSolver(lkh_max_trials=1)

                # load data and reference solutions from ``.txt`` file
                >>> solver.from_txt(file_path="examples/tsp/txt/tsp50_concorde.txt")
                
                # solve
                >>> solver.solve()
                    
                # Evaluate the quality of the solutions solved by LKH
                >>> solver.evaluate(calculate_gap=False)
                5.820372200519043
        """
        # check
        self._check_points_not_none()
        self._check_tours_not_none(ref=False)
        if calculate_gap:
            self._check_tours_not_none(ref=True)
            
        # variables
        points = self.ori_points if original else self.points
        tours = self.tours
        ref_tours = self.ref_tours

        # apply scale and dtype
        points = self._apply_scale_and_dtype(
            points=points, apply_scale=apply_scale,
            to_int=to_int, round_func=round_func
        )

        # prepare for evaluate
        tours_cost_list = list()
        samples = points.shape[0]
        if calculate_gap:
            ref_tours_cost_list = list()
            gap_list = list()

        # deal with different situation
        if tours.shape[0] != samples:
            # a problem has more than one solved tour
            tours = tours.reshape(samples, -1, tours.shape[-1])
            for idx in range(samples):
                evaluator = TSPEvaluator(points[idx], self.norm)
                solved_tours = tours[idx]
                solved_costs = list()
                for tour in solved_tours:
                    solved_costs.append(evaluator.evaluate(tour))
                solved_cost = np.min(solved_costs)
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(ref_tours[idx])
                    ref_tours_cost_list.append(ref_cost)
                    gap = (solved_cost - ref_cost) / ref_cost * 100
                    gap_list.append(gap)
        else:
            # a problem only one solved tour
            for idx in range(samples):
                evaluator = TSPEvaluator(points[idx], self.norm)
                solved_tour = tours[idx]
                solved_cost = evaluator.evaluate(solved_tour)
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(ref_tours[idx])
                    ref_tours_cost_list.append(ref_cost)
                    gap = (solved_cost - ref_cost) / ref_cost * 100
                    gap_list.append(gap)

        # calculate average cost/gap & std
        tours_costs = np.array(tours_cost_list)
        if calculate_gap:
            ref_costs = np.array(ref_tours_cost_list)
            gaps = np.array(gap_list)
        costs_avg = np.average(tours_costs)
        if calculate_gap:
            ref_costs_avg = np.average(ref_costs)
            gap_avg = np.sum(gaps) / samples
            gap_std = np.std(gaps)
            return costs_avg, ref_costs_avg, gap_avg, gap_std
        else:
            return costs_avg

    def solve(
        self,
        points: Union[np.ndarray, list] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        This method will be implemented in subclasses.
        
        :param points: np.ndarray, the coordinates of nodes. If given, the points 
            originally stored in the solver will be replaced.
        :param norm: boolean, the normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        
        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import TSPLKHSolver
                
                # create TSPLKHSolver
                >>> solver = TSPLKHSolver(lkh_max_trials=1)

                # load data and reference solutions from ``.tsp`` file
                >>> solver.from_tsplib(
                        tsp_file_path="examples/tsp/tsplib_1/problem/kroC100.tsp",
                        tour_file_path="examples/tsp/tsplib_1/solution/kroC100.opt.tour",
                        ref=False,
                        norm="EUC_2D",
                        normalize=True
                    )
                    
                # solve
                >>> solver.solve()
                [[ 0, 52, 39, 11, 48, 17, 28, 45, 23, 31, 60, 25,  6, 81, 77,  8,
                36, 15, 50, 62, 43, 65, 47, 83, 10, 51, 86, 95, 96, 80, 44, 32,
                99, 73, 56, 35, 13,  9, 91, 18, 98, 92,  3, 59, 68,  2, 72, 58,
                40, 88, 20, 22, 69, 75, 90, 93, 94, 49, 61, 82, 71, 85,  4, 42,
                55, 70, 37, 38, 27, 87, 97, 57, 33, 89, 24, 16,  7, 21, 74,  5,
                53,  1, 34, 67, 29, 76, 79, 64, 30, 46, 66, 54, 41, 19, 63, 78,
                12, 14, 26, 84,  0]]
        """
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "PCTSPSolver"