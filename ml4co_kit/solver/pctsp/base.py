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


import sys
import pickle
import numpy as np
from typing import Union
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.evaluate.pctsp import PCTSPEvaluator
from ml4co_kit.utils.time_utils import iterative_execution_for_file
from ml4co_kit.utils.type_utils import to_numpy, TASK_TYPE, SOLVER_TYPE


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
    :param ori_depots: :math:`(B \times 2)`, np.ndarray, the original coordinates of depots read.
    :param depots: :math:`(B \times 2)`, np.ndarray, the coordinates of depots.
    :param ori_points: :math:`(B\times N \times 2)`, np.ndarray, the original coordinates of points read.
    :param points: :math:`(B \times N \times 2)`, np.ndarray, the coordinates data called 
        by the solver during solving. They may initially be the same as ``ori_points``,
        but may later undergo standardization or scaling processing.
    :param penalties: :math:`(B \times N)`, np.ndarray, the penalties of nodes. 
    :param norm_prizes: :math:`(B\times N)`, np.ndarray, the prizes of nodes.
    :param tours: list, the solutions to the problems. 
    :param ref_tours: list, the reference solutions to the problems. 
    :param scale: int, magnification scale of coordinates. If the input coordinates are too large,
        you can scale them to 0-1 by setting ``normalize`` to True, and then use ``scale`` to adjust them.
        Note that the magnification scale only applies to ``points`` when solved by the solver.
    :param norm: string, coordinate type. It can be a 2D Euler distance or geographic data type.
    """
    def __init__(
        self, 
        solver_type: SOLVER_TYPE = None, 
        scale: int = 1e6,
        time_limit: float = 60.0,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(PCTSPSolver, self).__init__(
            task_type=TASK_TYPE.PCTSP, solver_type=solver_type, precision=precision
        )
        self.scale: np.ndarray = scale
        self.time_limit: float = time_limit
        self.nodes_num: int = None
        self.depots: np.ndarray = None
        self.ori_depots: np.ndarray = None
        self.points: np.ndarray = None
        self.ori_points: np.ndarray = None
        self.penalties: np.ndarray = None
        self.ori_penalties: np.ndarray = None
        self.norm_prizes: np.ndarray = None
        self.tours: list = None
        self.ref_tours: list = None
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

    def _check_ori_depots_dim(self):
        r"""
        Ensures that the ``ori_depots`` attribute is a 2D array. If ``ori_depots`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``ori_depots``
        is neither 1D nor 2D.
        """
        if self.ori_depots is not None:
            if self.ori_depots.ndim == 1:
                self.ori_depots = np.expand_dims(self.ori_depots, axis=0)
            if self.ori_depots.ndim != 2:
                raise ValueError("The dimensions of ``ori_depots`` cannot be larger than 2.")
    
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
            
    def _check_penalties_dim(self):
        r"""
        Ensures that the ``penalties`` attribute is a 2D array. If ``penalties`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``penalties``
        has more than 2 dimensions.
        """
        if self.penalties is not None:
            if self.penalties.ndim == 1:
                self.penalties = np.expand_dims(self.penalties, axis=0)
            if self.penalties.ndim != 2:
                raise ValueError("The dimensions of ``penalties`` cannot be larger than 2.")
            
    def _check_norm_prizes_dim(self):
        r"""
        Ensures that the ``norm_prizes`` attribute is a 2D array. If ``norm_prizes`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``norm_prizes``
        has more than 2 dimensions.
        """
        if self.norm_prizes is not None:
            if self.norm_prizes.ndim == 1:
                self.norm_prizes = np.expand_dims(self.norm_prizes, axis=0)
            if self.norm_prizes.ndim != 2:
                raise ValueError("The dimensions of ``norm_prizes`` cannot be larger than 2.")

    def _check_tours_dim(self):
        r"""
        Ensures that the ``tours`` attribute is a list of 1D arrays. If ``tours`` is a list
        of integers, it converts it to a list of arrays. Raises a ``ValueError`` if any
        element in ``tours`` is not a 1D array.
        """
        if self.tours is not None:
            if isinstance(self.tours[0], int):
                # if tours is a list of integers, convert it to a list of arrays
                self.tours = [np.array(self.tours)]
            try:
                self.tours = [np.array(tour) for tour in self.tours]
                assert all(tour.ndim == 1 for tour in self.tours)
            except (AssertionError, ValueError):
                raise ValueError("The ``tours`` must be a list of 1D arrays.")

    def _check_ref_tours_dim(self):
        r"""
        Ensures that the ``ref_tours`` attribute is a list of 1D arrays. If ``ref_tours``
        is a list of integers, it converts it to a list of arrays. Raises a ``ValueError``
        if any element in ``ref_tours`` is not a 1D array.
        """
        if self.ref_tours is not None:
            if isinstance(self.ref_tours[0], int):
                # if ref_tours is a list of integers, convert it to a list of arrays
                self.ref_tours = [np.array(self.ref_tours)]
            try:
                self.ref_tours = [np.array(tour) for tour in self.ref_tours]
                assert all(tour.ndim == 1 for tour in self.ref_tours)
            except (AssertionError, ValueError):
                raise ValueError("The ``ref_tours`` must be a list of 1D arrays.")
        
    def _check_depots_not_none(self):
        r"""
        Checks if the ``depots`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``depots`` is ``None``. 
        """
        if self.depots is None:
            message = (
                "``depots`` cannot be None! You can load the ``depots`` using the methods including "
                "``from_data``, ``from_txt`` or ``from_pkl``."
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
                "``from_data``, ``from_txt`` or ``from_pkl``."
            )
            raise ValueError(message)
        
    def _check_penalties_not_none(self):
        r"""
        Checks if the ``penalties`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``penalties`` is ``None``. 
        """
        if self.penalties is None:
            message = (
                "``penalties`` cannot be None! You can load the ``penalties`` "
                "using the methods including ``from_data``, ``from_txt`` or ``from_pkl``."
            )
            raise ValueError(message)
        
    def _check_norm_prizes_not_none(self):
        r"""
        Checks if the ``norm_prizes`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``norm_prizes`` is ``None``. 
        """
        if self.norm_prizes is None:
            message = (
                "``norm_prizes`` cannot be None! You can load the ``norm_prizes`` "
                "using the methods including ``from_data``, ``from_txt`` or ``from_pkl``."
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
            f"``{msg}`` cannot be None! You can use solvers based on ``PCTSPSolver``"
            "or use methods including ``from_data``, ``from_txt`` or ``from_pkl`` to obtain them."
        )  
        if ref:
            if self.ref_tours is None:
                raise ValueError(message)
        else:
            if self.tours is None:    
                raise ValueError(message)

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

    def _normalize_points_depots_penalties(self):
        r"""
        Normalizes the ``points`` attribute and ``depots`` attribute to scale 
        all coordinates between 0 and 1.
        """
        for idx in range(self.points.shape[0]):
            cur_points = self.points[idx]
            cur_depots = self.depots[idx]
            cur_penalties = self.penalties[idx]
            max_value = max(np.max(cur_points), np.max(cur_depots))
            min_value = min(np.min(cur_points), np.min(cur_depots))
            cur_points = (cur_points - min_value) / (max_value - min_value)
            cur_depots = (cur_depots - min_value) / (max_value - min_value)
            cur_penalties = cur_penalties / (max_value - min_value)
            self.points[idx] = cur_points
            self.depots[idx] = cur_depots
            self.penalties = cur_penalties

    def _check_demands_meet(self, ref: bool):
        r"""
        Checks if the ``tour`` satisfies the capacities demands. Raise a `ValueError` if 
        there is a split tour don't meet the demands.
        """
        tours = self.ref_tours if ref else self.tours
        num_tours = len(tours)
        num_instances = self.points.shape[0]
        if num_tours % num_instances != 0:
            raise ValueError(
                "The number of solutions cannot be divided evenly by the number of problems."
            )
        instance_per_sols = num_tours // num_tours
        for idx in range(num_tours):
            cur_prizes = self.norm_prizes[idx // instance_per_sols]
            cur_tour = tours[idx]
            cur_collect_prizes = np.sum(cur_prizes[cur_tour[1:-1] - 1])
            if cur_collect_prizes < 1 - 1e-5:
                message = (
                    f"Prize Constraint not met (ref = {ref}) in tour {idx}. "
                    f"The sum of the collected prizes is {cur_collect_prizes}."
                )
                raise ValueError(message)
        
    def _apply_scale_and_dtype(
        self, depots: np.ndarray, points: np.ndarray, penalties: np.ndarray, 
        apply_scale: bool, to_int: bool, round_func: str
    ):
        r"""
        Applies scaling and/or dtype conversion to the given ``points``.
        - Scales the points by ``self.scale`` if ``apply_scale`` is True.
        - Converts points to integers using the specified rounding function if ``to_int`` is True.
        """
        # apply scale
        if apply_scale:
            points = points * self.scale
            depots = depots * self.scale
            penalties = penalties * self.scale

        # dtype
        if to_int:
            round_func = self._get_round_func(round_func)
            points = round_func(points)
            depots = round_func(depots)
            penalties = round_func(penalties)
        
        return depots, points, penalties
    
    def _prepare_for_output(
        self, 
        depots: np.ndarray = None, 
        points: np.ndarray = None, 
        penalties: np.ndarray = None,
        tours: np.ndarray = None,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round"
    ):
        if points is None:
            return depots, points, penalties, tours

        # deal with different shapes
        samples = points.shape[0]
        if tours is not None and len(tours) != samples:
            # a problem has more than one solved tour
            if len(tours) % samples != 0:
                raise ValueError("The number of solutions cannot be divided evenly by the number of problems.")
            per_tour_num = len(tours) // samples    
            best_tour_list = list()
            for idx in range(samples):
                cur_eva = PCTSPEvaluator(
                    depots=depots[idx], 
                    points=points[idx], 
                    penalties=penalties[idx],
                    norm=self.norm
                )
                solved_tours = tours[idx*per_tour_num, (idx+1)*per_tour_num]
                best_tour = solved_tours[0]
                best_cost = cur_eva.evaluate(
                    route=best_tour, to_int=to_int, round_func=round_func
                )
                for tour in solved_tours[1:]:
                    cur_cost = cur_eva.evaluate(
                        route=tour, to_int=to_int, round_func=round_func
                    )
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_tour = tour
                best_tour_list.append(best_tour)
            tours = best_tour_list
        
        # apply scale and dtype
        depots, points, penalties = self._apply_scale_and_dtype(
            depots=depots, points=points, penalties=penalties, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )
        
        return depots, points, penalties, tours
    
    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time: bool = False
    ):
        r"""
        Read data from `.txt` file.

        :param file_path: string, path to the `.txt` file containing PCTSP instances data.
        :param ref: boolean, whether the solution is a reference solution.
        :param return_list: boolean, only use this function to obtain data, but do not save it to the solver. 
        :param norm: boolean, the normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import PCTSPSolver
                
                # create PCTSPSolver
                >>> solver = PCTSPSolver()

                # load data from ``.txt`` file
                >>> solver.from_txt(file_path="examples/pctsp/txt/pctsp50_concorde.txt")
                >>> solver.points.shape
                (16, 50, 2)
        """
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            # record to lists
            depots_list = list()
            points_list = list()
            penalties_list = list()
            norm_prizes_list = list()
            tours_list = list()
            
            # read by lines
            load_msg = f"Loading data from {file_path}"
            for line in iterative_execution_for_file(file, load_msg, show_time):
                # line to strings
                line = line.strip()
                split_line_0 = line.split("depots ")[1]
                split_line_1 = split_line_0.split(" points ")
                depots = split_line_1[0]
                split_line_2 = split_line_1[1].split(" penalties ")
                points = split_line_2[0]
                split_line_3 = split_line_2[1].split(" norm_prizes ")
                penalties = split_line_3[0]
                split_line_4 = split_line_3[1].split(" output ")
                norm_prizes = split_line_4[0]
                tours = split_line_4[1]
                
                # strings to array
                depots = depots.split(" ")
                depots = np.array([float(depots[0]), float(depots[1])], dtype=self.precision)
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ], dtype=self.precision
                )
                penalties = penalties.split(" ")
                penalties = np.array([
                    float(penalties[i]) for i in range(len(penalties))
                ], dtype=self.precision)
                norm_prizes = norm_prizes.split(" ")
                norm_prizes = np.array(
                    [float(norm_prizes[i]) for i in range(len(norm_prizes))], dtype=self.precision
                )
                tours = tours.split(" ")
                tours = np.array(
                    [int(tours[i]) for i in range(len(tours))]
                )
                
                # add to the list
                depots_list.append(depots)
                points_list.append(points)
                penalties_list.append(penalties)
                norm_prizes_list.append(norm_prizes)
                tours_list.append(tours)

        # check if return list
        if return_list:
            return (
                depots_list, points_list, penalties_list, \
                norm_prizes_list, tours_list
            )
            
        depots = np.array(depots_list)
        points = np.array(points_list)
        penalties = np.array(penalties_list)
        norm_prizes = np.array(norm_prizes_list)
        tours = tours_list

        # use ``from_data``
        self.from_data(
            depots=depots, points=points, penalties=penalties, 
            norm_prizes=norm_prizes, tours=tours, ref=ref, 
            norm=norm, normalize=normalize
        )

    def from_pickle(
        self,
        file_path: str,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
    ):
        # check the file format
        if not file_path.endswith(".pkl"):
            raise ValueError("Invalid file format. Expected a ``.pkl`` file.")
        
        # read the data from .pkl
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # check the data format
        if isinstance(data, list) and len(data) > 0:
            try:
                depots, points, penalties, norm_prizes, tours = zip(*data)
                depots = np.array(depots)
                points = np.array(points)
                penalties = np.array(penalties)
                norm_prizes = np.array(norm_prizes)
                self.from_data(
                    depots=depots, points=points, penalties=penalties, 
                    norm_prizes=norm_prizes, tours=tours, ref=ref, 
                    norm=norm, normalize=normalize
                )
            except Exception as e:
                raise ValueError(f"Invalid data format in PKL file: {e}")
        else:
            raise ValueError("PKL file should contain a list of tuples")

    def from_data(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        penalties: Union[list, np.ndarray] = None,
        norm_prizes: Union[list, np.ndarray] = None,
        tours: list = None,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
    ):
        """
        Read data from list or np.ndarray.

        :param depots: np.ndarray, the coordinates of depots. If given, the depots
            originally stored in the solver will be replaced.
        :param points: np.ndarray, the coordinates of nodes. If given, the points 
            originally stored in the solver will be replaced.
        :param penalties: np.ndarray, the penalties of nodes. If given, the penalties
            originally stored in the solver will be replaced.
        :param norm_prizes: np.ndarray, the prizes of nodes (normalization required).
            If given, the deterministic norm_prizes originally stored in the solver will be replaced.
        :param tours: np.ndarray, the solutions of the problems. If given, the tours
            originally stored in the solver will be replaced
        :param ref: boolean, whether the solution is a reference solution.
        :param norm: string, the normalization type for node coordinates (default is "EUC_2D").
        :param normalize: boolean, Whether to normalize node coordinates.

        .. dropdown:: Example

            :: 

                >>> import numpy as np
                >>> from ml4co_kit import PCTSPSolver
                
                # create PCTSPSolver
                >>> solver = PCTSPSolver()

                # load data from np.ndarray
                >>> solver.from_data(points=np.random.random(size=(10, 2)))
                >>> solver.points.shape
                (1, 10, 2)
        """
        # set norm
        self._set_norm(norm)
        
        # depots
        if depots is not None:
            depots = to_numpy(depots)
            self.ori_depots = depots
            self.depots = depots.astype(self.precision)
            self._check_depots_dim()

        # points
        if points is not None:
            points = to_numpy(points)
            self.ori_points = points
            self.points = points.astype(self.precision)
            self._check_ori_points_dim()
        
        # penalties
        if penalties is not None:
            penalties = to_numpy(penalties)
            self.ori_penalties = penalties
            self.penalties = penalties.astype(self.precision)
            self._check_penalties_dim()

        # normalize
        if normalize:
            self._normalize_points_depots_penalties()
    
        # norm_prizes
        if norm_prizes is not None:
            norm_prizes = to_numpy(norm_prizes)
            self.norm_prizes = norm_prizes.astype(self.precision)   
            self._check_norm_prizes_dim()
            
        # tours
        if tours is not None:
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
        r"""
        Output(store) data in ``txt`` format

        :param file_path: string, path to save the `.txt` file.
        :param original: boolean, whether to use ``original points`` or ``points``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.
         
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import PCTSPSolver
                
                # create PCTSPSolver
                >>> solver = PCTSPSolver()

                # load data from ``.pkl`` file
                >>> solver.from_pkl(file_path="examples/pctsp/txt/pctsp50.pkl")
                    
                # Output data in ``txt`` format
                >>> solver.to_txt(file_path="examples/pctsp/txt/pctsp50.txt")
        """
        # check
        self._check_depots_not_none()
        self._check_points_not_none()
        self._check_penalties_not_none()
        self._check_norm_prizes_not_none()
        self._check_tours_not_none(ref=False)
        
        # variables
        depots = self.depots
        points = self.ori_points if original else self.points
        penalties = self.penalties
        norm_prizes = self.norm_prizes
        tours = self.tours

        # deal with different shapes and apply scale and dtype
        depots, points, penalties, tours = self._prepare_for_output(
            depots=depots, points=points, penalties=penalties, tours=tours, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )
        
        # write
        with open(file_path, "w") as f:
            for idx in range(points.shape[0]):
                # write depot
                f.write(f"depots {str(depots[idx][0])} {str(depots[idx][1])} ")
                
                # write points
                f.write("points ")
                for node_x, node_y in points[idx]:
                    f.write(f"{str(node_x)} {str(node_y)} ")

                # write penalties
                f.write(f"penalties ")
                for i in range(len(penalties[idx])):
                    f.write(f"{str(penalties[idx][i])} ")

                # write norm_prizes
                f.write(f"norm_prizes ")
                for i in range(len(norm_prizes[idx])):
                    f.write(f"{str(norm_prizes[idx][i])} ")
                    
                # write tours
                f.write(f"output ")
                for node_idx in tours[idx]:
                    f.write(f"{node_idx} ")
                f.write("\n")
            f.close()
    
    def to_pickle(
        self,
        file_path: str = "example.pkl",
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round"
    ):
        r"""
        Output(store) data in ``pkl`` format

        :param file_path: string, path to save the `.pkl` file.
        :param original: boolean, whether to use ``original points`` or ``points``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.

        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import PCTSPSolver
                
                # create PCTSPSolver
                >>> solver = PCTSPSolver()

                # load data from ``.txt`` file
                >>> solver.from_txt(file_path="examples/pctsp/txt/pctsp50.txt")
                    
                # Output data in ``pickle`` format
                >>> solver.to_pickle(file_path="examples/pctsp/pkl/pctsp50_output.pkl")
        """
        # check
        self._check_depots_not_none()
        self._check_points_not_none()
        self._check_penalties_not_none()
        self._check_norm_prizes_not_none()
        self._check_tours_not_none(ref=False)
        
        # variables
        depots = self.ori_depots if original else self.depots
        points = self.ori_points if original else self.points
        penalties = self.ori_penalties if original else self.penalties
        norm_prizes = self.norm_prizes
        tours = self.tours

        # deal with different shapes and apply scale and dtype
        depots, points, penalties, tours = self._prepare_for_output(
            depots=depots, points=points, penalties=penalties, tours=tours, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )

        # write
        with open(file_path, "wb") as f:
            pickle.dump(
                list(zip(depots, points, penalties, norm_prizes, tours)), 
                f, pickle.HIGHEST_PROTOCOL
            )
        
    def evaluate(
        self,
        calculate_gap: bool = False,
        check_demands: bool = True,
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
            
                >>> from ml4co_kit import PCTSPORSolver
                
                # create PCTSPORSolver
                >>> solver = PCTSPORSolver()

                # load data and reference solutions from ``.txt`` file
                >>> solver.from_txt(file_path="examples/pctsp/txt/pctsp50.txt")
                
                # solve
                >>> solver.solve()
                    
                # Evaluate the quality of the solutions solved by ORTools
                >>> solver.evaluate(calculate_gap=False)
        """
        # check
        self._check_depots_not_none()
        self._check_points_not_none()
        self._check_penalties_not_none()
        self._check_norm_prizes_not_none()
        self._check_tours_not_none(ref=False)
        if check_demands:
            self._check_demands_meet(ref=False)
        if calculate_gap:
            self._check_tours_not_none(ref=True)
            if check_demands:
                self._check_demands_meet(ref=True)
            
        # variables
        depots = self.depots
        points = self.ori_points if original else self.points
        penalties = self.penalties
        tours = self.tours
        ref_tours = self.ref_tours

        # apply scale and dtype
        depots, points, penalties = self._apply_scale_and_dtype(
            depots=depots, points=points, penalties=penalties, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )

        # prepare for evaluate
        tours_cost_list = list()
        samples = points.shape[0]
        if calculate_gap:
            ref_tours_cost_list = list()
            gap_list = list()

        # deal with different situation
        if len(tours) != samples:
            if len(tours) % samples != 0:
                raise ValueError("The number of solutions cannot be divided evenly by the number of problems.")
            per_tour_num = len(tours) // samples
            for idx in range(samples):
                evaluator = PCTSPEvaluator(
                    depots=depots[idx], 
                    points=points[idx], 
                    penalties=penalties[idx],
                    norm=self.norm
                )
                solved_tours = tours[idx*per_tour_num, (idx+1)*per_tour_num]
                solved_costs = list()
                for tour in solved_tours:
                    solved_costs.append(evaluator.evaluate(
                        route=tour, to_int=to_int, round_func=round_func
                    ))
                solved_cost = np.min(solved_costs)
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(
                        route=ref_tours[idx], to_int=to_int, round_func=round_func
                    )
                    ref_tours_cost_list.append(ref_cost)
                    gap = (solved_cost - ref_cost) / ref_cost * 100
                    gap_list.append(gap)
        else:
            # a problem only one solved tour
            for idx in range(samples):
                evaluator = PCTSPEvaluator(
                    depots=depots[idx], 
                    points=points[idx], 
                    penalties=penalties[idx],
                    norm=self.norm
                )
                solved_cost = evaluator.evaluate(
                    route=tours[idx], to_int=to_int, round_func=round_func
                )
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(
                        route=ref_tours[idx], to_int=to_int, round_func=round_func
                    )
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
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        penalties: Union[list, np.ndarray] = None,
        norm_prizes: Union[list, np.ndarray] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        This method will be implemented in subclasses.
        
        :param depots: np.ndarray, the coordinates of depots. If given, the depots
            originally stored in the solver will be replaced.
        :param points: np.ndarray, the coordinates of nodes. If given, the points 
            originally stored in the solver will be replaced.
        :param penalties: np.ndarray, the penalties of nodes. If given, the penalties
            originally stored in the solver will be replaced.
        :param norm_prizes: np.ndarray, the deterministic norm_prizes of nodes.
            If given, the deterministic norm_prizes originally stored in the solver will be replaced.
        :param norm: boolean, the normalization type for node coordinates.
        :param normalize: boolean, whether to normalize node coordinates.
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        
        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import PCTSPORSolver
                
                # create 
                >>> solver = PCTSPORSolver()

                # load data and reference solutions from ``.pkl`` file
                >>> solver.from_pkl(file_path="examples/pctsp/pkl/pctsp50.pkl")
                    
                # solve
                >>> solver.solve()
        """
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "PCTSPSolver"