r"""
Basic solver for Asymmetric Traveling Salesman Problem (ATSP). 

ATSP is a variant of the classic TSP where the distance from 
one city to another may not be the same in both directions. 
It aims to find the shortest possible route that visits each city
once and returns to the starting point in a directed graph.
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
import math
import numpy as np
import networkx as nx
from typing import Union
from ml4co_kit.utils import tsplib95
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.evaluate.atsp.base import ATSPEvaluator
from ml4co_kit.utils.type_utils import to_numpy, TASK_TYPE, SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, iterative_execution_for_file


if sys.version_info.major == 3 and sys.version_info.minor == 8:
    from pyvrp.read import ROUND_FUNCS
else:
    from ml4co_kit.utils.round import ROUND_FUNCS


class ATSPSolver(SolverBase):
    r"""
    This class provides a basic framework for solving ATSP problems. It includes methods for 
    loading and outputting data in various file formats, normalizing points, and evaluating 
    solutions. Note that the actual solving method should be implemented in subclasses.
    
    :param nodes_num: :math:`N`, int, the number of nodes in ATSP problem.
    :param ori_dists: :math:`(B\times N\times N)`, np.ndarray, the original dists data read.
    :param dists: :math:`(B\times N\times N)`, np.ndarray, the dists data called by the solver 
        during solving. They may initially be the same as ``ori_dists``,
        but may later undergo standardization or scaling processing.
    :param tours: :math:`(B\times (N+1))`, np.ndarray, the solutions to the problems. 
    :param ref_tours: :math:`(B\times (N+1))`, np.ndarray, the reference solutions to the problems. 
    :param scale: int, magnification scale of coordinates. If the input coordinates are too large,
        you can scale them to 0-1 by setting ``normalize`` to True, and then use ``scale`` to adjust them.
        Note that the magnification scale only applies to ``points`` when solved by the solver.
    """
    def __init__(self, solver_type: SOLVER_TYPE = None, scale: int = 1e6):
        super(ATSPSolver, self).__init__(
            task_type=TASK_TYPE.ATSP, solver_type=solver_type
        )
        self.scale = scale
        self.dists: np.ndarray = None
        self.ori_dists: np.ndarray= None
        self.tours: np.ndarray = None
        self.ref_tours: np.ndarray = None
        self.nodes_num: int = None

    def _check_dists_dim(self):
        r"""
        Ensures that the ``dists`` attribute is a 3D array. If ``dists`` is a 2D array,
        it adds an additional dimension to make it 3D. Raises a ``ValueError`` if ``dists`` 
        is neither 2D or 3D. Also sets the ``nudes_norm`` attribute to the number of nodes
        (points) in the problem. 
        """
        if self.dists is not None:
            if self.dists.ndim == 2:
                self.dists = np.expand_dims(self.dists, axis=0)
            if self.dists.ndim != 3:
                raise ValueError("``dists`` must be a 2D or 3D array.")
            self.nodes_num = self.dists.shape[-1]

    def _check_ori_dists_dim(self):
        r"""
        Ensures that the ``ori_dists`` attribute is a 3D array. Calls ``_check_dists_dim``
        to validate the ``dists`` attribute first. If ``ori_dists`` is a 2D array, it adds
        an additional dimension to make it 3D. Raises a ``ValueError`` if ``ori_dists`` is
        neither 2D nor 3D.
        """
        self._check_dists_dim()
        if self.ori_dists is not None:
            if self.ori_dists.ndim == 2:
                self.ori_dists = np.expand_dims(self.ori_dists, axis=0)
            if self.ori_dists.ndim != 3:
                raise ValueError("The ``ori_dists`` must be 2D or 3D array.")

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

    def _check_dists_not_none(self):
        r"""
        Checks if the ``dists`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``dists`` is ``None``. 
        """
        if self.dists is None:
            message = (
                "``dists`` cannot be None! You can load the dists using the methods"
                "``from_data``, ``from_txt``, ``from_atsp`` or ``from_atsp_folder``."
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
            f"``{msg}`` cannot be None! You can use solvers based on "
            "``ATSPSolver`` like ``ATSPLKHSolver`` or use methods including "
            "``from_data``, ``from_txt`` or ``from_tsplib`` to obtain them."
        )  
        if ref:
            if self.ref_tours is None:
                raise ValueError(message)
        else:
            if self.tours is None:    
                raise ValueError(message)

    def _normalize_dists(self):
        r"""
        Normalizes the ``dists`` attribute to scale all dists between 0 and 1.
        """
        for idx in range(self.dists.shape[0]):
            cur_dists = self.dists[idx]
            max_value = np.max(cur_dists)
            min_value = np.min(cur_dists)
            cur_dists = (cur_dists - min_value) / (max_value - min_value)
            self.dists[idx] = cur_dists

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
        self, dists: np.ndarray, apply_scale: bool, to_int: bool, round_func: str
    ):
        r"""
        Applies scaling and/or dtype conversion to the given ``dists``.
        - Scales the dists by ``self.scale`` if ``apply_scale`` is True.
        - Converts dists to integers using the specified rounding function if ``to_int`` is True.
        """
        # apply scale
        if apply_scale:
            dists = dists * self.scale

        # dtype
        if to_int:
            round_func = self._get_round_func(round_func)
            dists = round_func(dists)
        
        return dists

    def _read_data_from_atsp_file(self, atsp_file_path: str) -> np.ndarray:
        r"""
        Reads ATSP dists from a TSPLIB file.
        """
        tsplib_data = tsplib95.load(atsp_file_path)
        try:
            dists = nx.to_numpy_array(tsplib_data.get_graph())
        except:
            try:
                dists = np.array(tsplib_data.edge_weights)
            except:
                raise RuntimeError("Error in loading {}".format(atsp_file_path))
        return dists

    def _read_tour_from_tour_file(self, tour_file_path: str) -> np.ndarray:
        r"""
        Reads an ATSP tour from a TSPLIB tour file.
        """
        tsp_tour = tsplib95.load(tour_file_path)
        tsp_tour = tsp_tour.tours
        tsp_tour: list
        tsp_tour = tsp_tour[0]
        tsp_tour.append(1)
        tour = np.array(tsp_tour) - 1
        return tour
    
    def from_tsplib(
        self, 
        atsp_file_path: str = None,
        tour_file_path: str = None,
        ref: bool = False,
        normalize: bool = False
    ):
        """
        Read data from single TSPLIB type file
        
        :param atsp_file_path: string, path to the `.tsp` or `.atsp` file containing ATSP instance data.
            If given, the solver will read dists adjacency matrix from the file.
        :param tour_file_path: string, path to the `.tour` file containing ATSP solution data.
            If given, the solver will read tour from the file.
        :param ref: boolean, whether the solution is a reference solution.
        :param normalize: boolean, whether to normalize node coordinates.

        .. note::
            - If the given ``atsp_file_path`` does not end with ``.tsp`` or ``.atsp``, the solver will raise ``ValueError``.
            - If the given ``tour_file_path`` does not end with ``.tour`` or ``.opt_tour``, the solver will raise ``ValueError``.
           
        .. dropdown:: Example

            ::   
            
                >>> from ml4co_kit import ATSPSolver
                
                # create ATSPSolver
                >>> solver = ATSPSolver()

                # load data from ``.tsp`` or ``.atsp``, '.tour' or ``.opt.tour`` files
                >>> solver.from_tsplib(
                        atsp_file_path="examples/atsp/tsplib_1/problem/gr24.tsp",
                        tour_file_path="examples/atsp/tsplib_1/solution/gr24.opt.tour",
                        ref=False,
                        normalize=False
                    )
                >>> solver.dists.shape
                (1, 25, 12)
                >>> solver.tours.shape
                (1, 25)
                >>> solver.dists[0][0]
                [  0. 257. 187.  91. 150.  80. 130. 134. 243. 185. 214.  70. 272. 219.
                293.  54. 211. 290. 268. 261. 175. 250. 192. 121.]
                
                # If you want to normalize the input data (for easy machine learning method calls), 
                # you can set ``normalize`` to True.
                >>> solver.from_tsplib(
                        atsp_file_path="examples/atsp/tsplib_1/problem/gr24.tsp",
                        tour_file_path="examples/atsp/tsplib_1/solution/gr24.opt.tour",
                        ref=False,
                        normalize=True
                    )
                >>> solver.dists[0][0]
                [0.         0.6606684  0.4807198  0.23393317 0.3856041  0.20565553
                0.33419022 0.344473   0.6246787  0.4755784  0.5501285  0.17994858
                0.69922876 0.562982   0.75321335 0.13881747 0.54241645 0.7455013
                0.688946   0.6709511  0.44987145 0.6426735  0.49357328 0.311054  ]
        """
        # init
        dists = None
        tour = None
        
        # read dists from .tsp file
        if atsp_file_path is not None:
            if not atsp_file_path.endswith(".atsp") and not atsp_file_path.endswith(".tsp"):
                raise ValueError("Invalid file format. Expected a ``.atsp`` or ``.tsp`` file.")
            dists = self._read_data_from_atsp_file(atsp_file_path)
        
        # read tour from .tour file
        if tour_file_path is not None:
            if not tour_file_path[-5:] == ".tour":
                raise ValueError(
                    "Invalid file format. Expected a ``.tour`` or ``.opt.tour`` file."
                )
            tour = self._read_tour_from_tour_file(tour_file_path)
        
        # use ``from_data``
        self.from_data(
            dists=dists, tours=tour, ref=ref, normalize=normalize
        )

    def from_tsplib_folder(
        self, 
        atsp_folder_path: str = None,
        tour_folder_path: str = None,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time: bool = False
    ):
        """
        Read data from the folder containing TSPLIB type data

        :param atsp_folder_path: string, path to the folder containing `.tsp` or `.atsp` files.
            If given, the solver will read dists matrix from the folder.
        :param tour_folder_path: string, path to the folder containing `.tour` files.
            If given, the solver will read tour from the folder.
        :param ref: boolean, whether the solution is a reference solution.
        :param return_list: boolean, only use this function to obtain data, but do not save it to the solver. 
        :param norm: string, the normalization type for dists matrix.
        :param normalize: boolean, whether to normalize dists matrix.
        :param show_time: boolean, whether the data is being read with a visual progress display.
        
        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import ATSPSolver
                
                # create ATSPSolver
                >>> solver = ATSPSolver()

                # load data from the tsplib folder
                >>> solver.from_tsplib_folder(
                        atsp_folder_path="examples/atsp/tsplib_2/problem",
                        tour_folder_path="examples/atsp/tsplib_2/solution"
                    )
                >>> solver.dists.shape
                (1, 25, 12)
                >>> 
                (1, 25)

                # When the number of nodes is not consistent, ``return_list`` can be used to return data.
                >>> dists_flag, tours_list = solver.from_tsplib_folder(
                        atsp_folder_path="examples/atsp/tsplib_1/problem",
                        tour_folder_path="examples/atsp/tsplib_1/solution",
                        return_list=True
                    )
                >>> dists_flag[0].shape
                (25, 12)
        """
        # init
        dists = None
        tours = None
        dists_flag = False if atsp_folder_path is None else True
        tours_flag = False if tour_folder_path is None else True
        
        # only dists
        if dists_flag and not tours_flag:
            dists_list = list()
            files = os.listdir(atsp_folder_path)
            files.sort()
            load_msg = f"Loading data from {atsp_folder_path}"
            for file_name in iterative_execution_for_file(files, load_msg, show_time):
                atsp_file_path = os.path.join(atsp_folder_path, file_name)
                if not atsp_file_path.endswith(".atsp")  and not atsp_file_path.endswith(".tsp"):
                    continue
                dists = self._read_data_from_atsp_file(atsp_file_path)
                dists_list.append(dists)

        # only tours
        if not dists_flag and tours_flag:
            tours_list = list()
            files = os.listdir(tour_folder_path)
            files.sort()
            load_msg = f"Loading solutions from {tour_folder_path}"
            for file_name in iterative_execution_for_file(files, load_msg, show_time):
                tour_file_path = os.path.join(tour_folder_path, file_name)
                if not tour_file_path[-5:] == ".tour":
                    continue
                tour = self._read_tour_from_tour_file(tour_file_path)
                tours_list.append(tour)
        
        # both dists and tours [must have the same filename]
        if dists_flag and tours_flag:
            dists_list = list()
            tours_list = list()
            files = os.listdir(atsp_folder_path)
            files.sort()
            load_msg = f"Loading data from {atsp_folder_path} and solutions from {tour_folder_path}"
            for file_name in iterative_execution_for_file(files, load_msg, show_time):
                # dists
                atsp_file_path = os.path.join(atsp_folder_path, file_name)
                if not atsp_file_path.endswith(".atsp") and not atsp_file_path.endswith(".tsp"):
                    continue
                dists = self._read_data_from_atsp_file(atsp_file_path)
                dists_list.append(dists)
                # tour
                tour_file_path = os.path.join(
                    tour_folder_path, file_name.replace(".tsp", ".opt.tour")
                )
                tour = self._read_tour_from_tour_file(tour_file_path)
                tours_list.append(tour)
                
        # return list
        if return_list:
            if dists_flag:
                if tours_flag:
                    return dists_list, tours_list
                else:
                    return dists_list
            else:
                if tours_flag:
                    return tours_list
        
        # check
        message = (
            "This method does not support instances of different numbers of nodes. "
            "If you want to read the data, please set ``return_list`` as True. "
            "Anyway, the data will not be saved in the solver. "
            "Please convert the data to ``np.ndarray`` externally before calling the solver."
        )
        if dists_flag:
            try:
                dists = np.array(dists_list)
            except Exception as e:
                raise Exception(message) from e
        if tours_flag:
            try:
                tours = np.array(tours_list)
            except Exception as e:
                raise Exception(message) from e
        
        # use ``from_data``
        self.from_data(
            dists=dists, tours=tours, ref=ref, normalize=normalize
        )

    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        normalize: bool = False,
        show_time = False
    ):
        """
        Read data from `.txt` file.

        :param file_path: string, path to the `.txt` file containing ATSP instances data.
        :param ref: boolean, whether the solution is a reference solution.
        :param return_list: boolean, only use this function to obtain data, but do not save it to the solver. 
        :param norm: boolean, the normalization type for dists matrix.
        :param normalize: boolean, whether to normalize dists matrix.
        :param show_time: boolean, whether the data is being read with a visual progress display.

        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import ATSPSolver
                
                # create ATSPSolver
                >>> solver = ATSPSolver()

                # load data from ``.txt`` file 
                >>> solver.from_txt(file_path="examples/atsp/txt/atsp50.txt")
                >>> solver.dists.shape
                (16, 50, 50)
                >>> solver.tours.shape
                (16, 51)              
        """
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            dists_list = list()
            tour_list = list()
            load_msg = f"Loading data from {file_path}"
            for line in iterative_execution_for_file(file, load_msg, show_time):
                line = line.strip()
                split_line = line.split(" output ")
                dist = split_line[0]
                tour = split_line[1]
                tour = tour.split(" ")
                tour = np.array([int(t) for t in tour])
                tour -= 1
                tour_list.append(tour)
                dist = dist.split(" ")
                dist.append('')           
                dist = np.array([float(dist[2*i]) for i in range(len(dist) // 2)])
                num_nodes = int(math.sqrt(len(dist)))
                dist = dist.reshape(num_nodes, num_nodes)
                dists_list.append(dist)

        if return_list:
            return dists_list, tour_list

        try:
            dists = np.array(dists_list)
            tours = np.array(tour_list)
        except Exception as e:
            message = (
                "This method does not support instances of different numbers of nodes. "
                "If you want to read the data, please set ``return_list`` as True. "
                "Anyway, the data will not be saved in the solver. "
                "Please convert the data to ``np.ndarray`` externally before calling the solver."
            )
            raise Exception(message) from e

        self.from_data(
            dists=dists, tours=tours, ref=ref, normalize=normalize
        )

    def from_data(
        self, 
        dists: Union[list, np.ndarray] = None,
        tours: Union[list, np.ndarray] = None,
        ref: bool = False,
        normalize: bool = False,
    ):
        """
        Read data from list or np.ndarray

        :param dists: list or np.ndarray, the dists matrix. If given, the matrix
            originally stored in the solver will be replaced.
        :param tours: np.ndarray, the solutions of the problems. If given, the tours
            originally stored in the solver will be replaced.
        :param ref: boolean, whether the solution is a reference solution.
        :param norm: string, the normalization type for dists matrix (default is "EUC_2D").
        :param normalize: boolean, Whether to normalize the dists.

        .. dropdown:: Example

            ::
            
                >>> import numpy as np
                >>> from ml4co_kit import ATSPSolver
                
                # create ATSPSolver
                >>> solver = ATSPSolver()

                # load data from np.ndarray
                >>> solver.from_data(dists=np.random.random(size=(10, 10)))
                >>> solver.dists.shape
                (1, 10, 10)

        """
        # dists
        if dists is not None:
            dists = to_numpy(dists)
            self.ori_dists = dists
            self.dists = dists.astype(np.float32)
            self._check_ori_dists_dim()
            if normalize:
                self._normalize_dists()

        # tours
        if tours is not None:
            tours = to_numpy(tours).astype(np.int32)
            if ref:
                self.ref_tours = tours
                self._check_ref_tours_dim()
            else:
                self.tours = tours
                self._check_tours_dim()
    
    def to_tsplib_folder(
        self,
        atsp_save_dir: str = None,
        atsp_filename: str = None,
        tour_save_dir: str = None,
        tour_filename: str = None,
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round",
        show_time: bool = False
    ):
        """
        Output(store) data in ``atsp`` format

        :param atsp_save_path: string, path to save the `.tsp` files. If given, 
            the coordinates will be saved as ``.atsp`` file for each instance.
        :param atsp_filename: string, the basic file name of the `.atsp` files.
        :param tour_save_path: string, path to save the `.opt.tour` files. If given,
            the solution will be saved as ``.opt.tour`` file for each instance.
        :param tour_filename: string, the basic file name of the `.opt.tour` files.
        :param original: boolean, whether to use ``original points`` or ``points``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.
        :param show_time: boolean, whether the data is being output with a visual progress display.

        .. note::
            ``points`` and ``tours`` must not be None.
         
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import ATSPSolver
                
                # create ATSPSolver
                >>> solver = ATSPSolver()

                # load data from ``.txt`` file
                >>> solver.from_txt(file_path="examples/atsp/txt/atsp50.txt")
                    
                # Output data in TSPLIB format
                >>> solver.to_tsplib_folder(
                        tsp_save_dir="atsp50/problem",
                        tsp_filename="atsp50",
                        tour_save_dir="atsp50/solution",
                        tour_filename="atsp50"
                    )
        """
        # .atsp files
        if atsp_save_dir is not None:
            # preparation
            if atsp_filename.endswith(".atsp"):
                atsp_filename = atsp_filename.replace(".atsp", "")
            self._check_dists_not_none()
            dists = self.ori_dists if original else self.dists
            samples = dists.shape[0]

            # apply scale and dtype
            dists = self._apply_scale_and_dtype(
                dists=dists, apply_scale=apply_scale,
                to_int=to_int, round_func=round_func
            )

            # makedirs
            if not os.path.exists(atsp_save_dir):
                os.makedirs(atsp_save_dir)

            # write
            write_msg = f"Writing tsp files to {atsp_save_dir}"
            for idx in iterative_execution(range, samples, write_msg, show_time):
                # file name & save path
                if samples == 1:
                    name = atsp_filename + f".atsp"
                else:
                    name = atsp_filename + f"-{idx}.atsp"
                save_path = os.path.join(atsp_save_dir, name)
                with open(save_path, "w") as f:
                    f.write(f"NAME : {name}\n")
                    f.write(f"COMMENT : Generated by ML4CO-Kit\n")
                    f.write("TYPE : ATSP\n")
                    f.write(f"DIMENSION : {self.nodes_num}\n")
                    f.write(f"EDGE_WEIGHT_TYPE : EXPLICIT\n")
                    f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
                    f.write("EDGE_WEIGHT_SECTION:\n")
                    for i in range(self.nodes_num):
                        line = ' '.join([str(elem) for elem in dists[idx][i]])
                        f.write(f"{line}\n")
                    f.write("EOF\n")

        # .opt.tour files
        if tour_save_dir is not None:
            # preparation
            if tour_filename.endswith(".opt.tour"):
                tour_filename = tour_filename.replace(".opt.tour", "")
            if tour_filename.endswith(".tour"):
                tour_filename = tour_filename.replace(".tour", "")
            self._check_tours_not_none(ref=False)
            tours = self.tours
            samples = tours.shape[0]
            
            # makedirs
            if not os.path.exists(tour_save_dir):
                os.makedirs(tour_save_dir)

            # write
            write_msg = f"Writing tour files to {tour_save_dir}"
            for idx in iterative_execution(range, samples, write_msg, show_time):
                if samples == 1:
                    name = tour_filename + f".opt.tour"
                else:
                    name = tour_filename + f"-{idx}.opt.tour"
                save_path = os.path.join(tour_save_dir, name)
                with open(save_path, "w") as f:
                    f.write(f"NAME: {name} Solved by ML4CO-Kit\n")
                    f.write(f"TYPE: TOUR\n")
                    f.write(f"DIMENSION: {self.nodes_num}\n")
                    f.write(f"TOUR_SECTION\n")
                    for i in range(self.nodes_num):
                        f.write(f"{tours[idx][i]}\n")
                    f.write(f"-1\n")
                    f.write(f"EOF\n")

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
        :param tour_file_path: string, path to the `.tour` file containing ATSP solution data.
            if given, the solver will read tour from the file.
        :param original: boolean, whether to use ``original dists`` or ``dists``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.

        .. note::
            ``points`` and ``tours`` must not be None.
         
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import ATSPSolver
                
                # create ATSPSolver
                >>> solver = ATSPSolver()

                # load data from ``.tsp`` or ``.atsp`` and ``.opt.tour`` files
                >>> solver.from_tsplib(
                        atsp_file_path="examples/atsp/tsplib_1/problem/gr24.tsp",
                        tour_file_path="examples/atsp/tsplib_1/solution/gr24.opt.tour",
                        ref=False,
                        normalize=True
                    )
                    
                # Output data in ``txt`` format
                >>> solver.to_txt("gr24.txt")
        """
        # check
        self._check_dists_not_none()
        self._check_tours_not_none(ref=False)
        
        # variables
        dists = self.ori_dists if original else self.dists
        tours = self.tours

        # deal with different shapes
        samples = dists.shape[0]
        if tours.shape[0] != samples:
            # a problem has more than one solved tour
            samples_tours = tours.reshape(samples, -1, tours.shape[-1])
            best_tour_list = list()
            for idx, solved_tours in enumerate(samples_tours):
                cur_eva = ATSPEvaluator(dists[idx])
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
        dists = self._apply_scale_and_dtype(
            dists=dists, apply_scale=apply_scale,
            to_int=to_int, round_func=round_func
        )


        # write
        with open(file_path, "w") as f:
            for dist, tour in zip(dists, tours):
                dist: np.ndarray = dist.reshape(-1)
                f.write(" ".join(str(x) + str(" ") for x in dist))
                f.write(str("output") + str(" "))
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
        :param original: boolean, whether to use ``original dists`` or ``dists``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.

        .. note::
            - Please make sure the ``dists`` and the ``tours`` are not None.
            - If you set the ``calculate_gap`` as True, please make sure the ``ref_tours`` is not None.
        
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import ATSPLKHSolver
                
                # create ATSPLKHSolver
                >>> solver = ATSPLKHSolver(lkh_max_trials=1)

                # load data and reference solutions from ``.tsp`` file
                >>> solver.from_tsplib(
                        atsp_file_path="examples/atsp/tsplib_1/problem/gr24.tsp",
                        tour_file_path="examples/atsp/tsplib_1/solution/gr24.opt.tour",
                        ref=False,
                        normalize=True
                    )
                    
                # solve
                >>> solver.solve()
                    
                # Evaluate the quality of the solutions solved by LKH
                >>> solver.evaluate(calculate_gap=False)
                273.0
        """
        # check
        self._check_dists_not_none()
        self._check_tours_not_none(ref=False)
        if calculate_gap:
            self._check_tours_not_none(ref=True)
            
        # variables
        dists = self.ori_dists if original else self.dists
        tours = self.tours
        ref_tours = self.ref_tours

        # apply scale and dtype
        dists = self._apply_scale_and_dtype(
            dists=dists, apply_scale=apply_scale,
            to_int=to_int, round_func=round_func
        )

        # prepare for evaluate
        tours_cost_list = list()
        samples = dists.shape[0]
        if calculate_gap:
            ref_tours_cost_list = list()
            gap_list = list()

        # deal with different situation
        if tours.shape[0] != samples:
            # a problem has more than one solved tour
            tours = tours.reshape(samples, -1, tours.shape[-1])
            for idx in range(samples):
                evaluator = ATSPEvaluator(dists[idx])
                solved_tours = tours[idx]
                solved_costs = list()
                for tour in solved_tours:
                    cost = evaluator.evaluate(
                        route=tour,
                        to_int=to_int, 
                        round_func=round_func
                    )
                    solved_costs.append(cost)
                solved_cost = np.min(solved_costs)
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(
                        route=ref_tours[idx], 
                        to_int=to_int, 
                        round_func=round_func
                    )
                    ref_tours_cost_list.append(ref_cost)
                    gap = (solved_cost - ref_cost) / ref_cost * 100
                    gap_list.append(gap)
        else:
            # a problem only one solved tour
            for idx in range(samples):
                evaluator = ATSPEvaluator(dists[idx])
                solved_tour = tours[idx]
                solved_cost = evaluator.evaluate(
                    route=solved_tour,
                    to_int=to_int, 
                    round_func=round_func
                )
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(
                        route=ref_tours[idx], 
                        to_int=to_int, 
                        round_func=round_func
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
        dists: Union[np.ndarray, list] = None,
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        This method will be implemented in subclasses.
        
        :param dists: np.ndarray, the dist matrix. If given, the dists 
            originally stored in the solver will be replaced.
        :param normalize: boolean, whether to normalize the dists.
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.

        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import ATSPLKHSolver
                
                # create ATSPLKHSolver
                >>> solver = ATSPLKHSolver(lkh_max_trials=1)

                # load data and reference solutions from ``.atsp`` file
                >>> solver.from_tsplib(
                        atsp_file_path="examples/atsp/tsplib_1/problem/ft53.tsp",
                        ref=False,
                        normalize=True
                    )
                    
                # solve
                >>> solver.solve()
                [[ 0  4  2 17 16 15 37 39 38 36 35 40 21 20 24 23 22 19 18  1  8  9  7  6
                5 51 49 52 50 48 29 28 25 27 26  3 13 11 10 12 14 41 47 42 46 43 45 44
                34 32 33 31 30  0]]
         """
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "ATSPSolver"