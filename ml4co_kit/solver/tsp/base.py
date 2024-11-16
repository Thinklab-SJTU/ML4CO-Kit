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


class TSPSolver(SolverBase):
    def __init__(self, solver_type: SOLVER_TYPE = None, scale: int = 1e6):
        super(TSPSolver, self).__init__(
            task_type=TASK_TYPE.TSP, solver_type=solver_type
        )
        self.scale: np.ndarray = scale
        self.points: np.ndarray = None
        self.ori_points: np.ndarray = None
        self.tours: np.ndarray = None
        self.ref_tours: np.ndarray = None
        self.nodes_num: int = None
        self.norm: str = None
        
    def check_points_dim(self):
        if self.points is not None:
            if self.points.ndim == 2:
                self.points = np.expand_dims(self.points, axis=0)
            if self.points.ndim != 3:
                raise ValueError("``points`` must be a 2D or 3D array.")
            self.nodes_num = self.points.shape[1]

    def check_ori_points_dim(self):
        self.check_points_dim()
        if self.ori_points is not None:
            if self.ori_points.ndim == 2:
                self.ori_points = np.expand_dims(self.ori_points, axis=0)
            if self.ori_points.ndim != 3:
                raise ValueError("The ``ori_points`` must be 2D or 3D array.")

    def check_tours_dim(self):
        if self.tours is not None:
            if self.tours.ndim == 1:
                self.tours = np.expand_dims(self.tours, axis=0)
            if self.tours.ndim != 2:
                raise ValueError("The dimensions of ``tours`` cannot be larger than 2.")

    def check_ref_tours_dim(self):
        if self.ref_tours is not None:
            if self.ref_tours.ndim == 1:
                self.ref_tours = np.expand_dims(self.ref_tours, axis=0)
            if self.ref_tours.ndim != 2:
                raise ValueError(
                    "The dimensions of the ``ref_tours`` cannot be larger than 2."
                )

    def check_points_not_none(self):
        if self.points is None:
            message = (
                "``points`` cannot be None! You can load the ``points`` using the methods including "
                "``from_data``, ``from_txt``, ``from_tsplib`` or ``from_tsplib_folder`."
            )
            raise ValueError(message)

    def check_tours_not_none(self, ref: bool):
        msg = "ref_tours" if ref else "tours"
        message = (
            f"``{msg}`` cannot be None! You can use solvers based on ``TSPSolver``"
            "like ``TSPLKHSolver`` or use methods including ``from_data``, "
            "``from_txt``, ``from_tsplib`` or ``from_tsplib_folder`` to obtain them."
        )  
        if ref:
            if self.ref_tours is None:
                raise ValueError(message)
        else:
            if self.tours is None:    
                raise ValueError(message)

    def set_norm(self, norm: str):
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

    def normalize_points(self):
        for idx in range(self.points.shape[0]):
            cur_points = self.points[idx]
            max_value = np.max(cur_points)
            min_value = np.min(cur_points)
            cur_points = (cur_points - min_value) / (max_value - min_value)
            self.points[idx] = cur_points

    def get_round_func(self, round_func: str):
        if (key := str(round_func)) in ROUND_FUNCS:
            round_func = ROUND_FUNCS[key]
        if not callable(round_func):
            raise TypeError(
                f"round_func = {round_func} is not understood. Can be a function,"
                f" or one of {ROUND_FUNCS.keys()}."
            )
        return round_func

    def apply_scale_and_dtype(
        self, points: np.ndarray, apply_scale: bool, to_int: bool, round_func: str
    ):
        # apply scale
        if apply_scale:
            points = points * self.scale

        # dtype
        if to_int:
            round_func = self.get_round_func(round_func)
            points = round_func(points)
        
        return points

    def _read_data_from_tsp_file(self, tsp_file_path: str) -> np.ndarray:
        tsplib_data = tsplib95.load(tsp_file_path)
        points = np.array(list(tsplib_data.node_coords.values()))
        if points is None:
            raise RuntimeError("Error in loading {}".format(tsp_file_path))
        return points

    def _read_tour_from_tour_file(self, tour_file_path: str) -> np.ndarray:
        tsp_tour = tsplib95.load(tour_file_path)
        tsp_tour = tsp_tour.tours
        tsp_tour: list
        tsp_tour = tsp_tour[0]
        tsp_tour.append(1)
        tour = np.array(tsp_tour) - 1
        return tour

    def from_tsplib(
        self, 
        tsp_file_path: str = None,
        tour_file_path: str = None,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False
    ):
        # init
        points = None
        tour = None
        
        # read points from .tsp file
        if tsp_file_path is not None:
            if not tsp_file_path.endswith(".tsp"):
                raise ValueError("Invalid file format. Expected a ``.tsp`` file.")
            points = self._read_data_from_tsp_file(tsp_file_path)
        
        # read tour from .tour file
        if tour_file_path is not None:
            if not tour_file_path[-5:] == ".tour":
                raise ValueError(
                    "Invalid file format. Expected a ``.tour`` or ``.opt.tour`` file."
                )
            tour = self._read_tour_from_tour_file(tour_file_path)
        
        # use ``from_data``
        self.from_data(
            points=points, tours=tour, ref=ref, norm=norm, normalize=normalize
        )
    
    def from_tsplib_folder(
        self, 
        tsp_folder_path: str = None,
        tour_folder_path: str = None,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time: bool = False
    ):
        # init
        points = None
        tours = None
        points_flag = False if tsp_folder_path is None else True
        tours_flag = False if tour_folder_path is None else True
        
        # only points
        if points_flag and not tours_flag:
            points_list = list()
            files = os.listdir(tsp_folder_path)
            files.sort()
            load_msg = f"Loading data from {tsp_folder_path}"
            for file_name in iterative_execution_for_file(files, load_msg, show_time):
                tsp_file_path = os.path.join(tsp_folder_path, file_name)
                if not tsp_file_path.endswith(".tsp"):
                    continue
                points = self._read_data_from_tsp_file(tsp_file_path)
                points_list.append(points)

        # only tours
        if not points_flag and tours_flag:
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
        
        # both points and tours [must have the same filename]
        if points_flag and tours_flag:
            points_list = list()
            tours_list = list()
            files = os.listdir(tsp_folder_path)
            files.sort()
            load_msg = f"Loading data from {tsp_folder_path} and solutions from {tour_folder_path}"
            for file_name in iterative_execution_for_file(files, load_msg, show_time):
                # points
                tsp_file_path = os.path.join(tsp_folder_path, file_name)
                if not tsp_file_path.endswith(".tsp"):
                    continue
                points = self._read_data_from_tsp_file(tsp_file_path)
                points_list.append(points)
                # tour
                tour_file_path = os.path.join(
                    tour_folder_path, file_name.replace(".tsp", ".opt.tour")
                )
                tour = self._read_tour_from_tour_file(tour_file_path)
                tours_list.append(tour)
                
        # return list
        if return_list:
            if points_flag:
                if tours_flag:
                    return points_list, tours_list
                else:
                    return points_list
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
        if points_flag:
            try:
                points = np.array(points_list)
            except Exception as e:
                raise Exception(message) from e
        if tours_flag:
            try:
                tours = np.array(tours_list)
            except Exception as e:
                raise Exception(message) from e
        
        # use ``from_data``
        self.from_data(
            points=points, tours=tours, ref=ref, norm=norm, normalize=normalize
        )        
        
    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time = False
    ):
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
        # set norm
        self.set_norm(norm)
    
        # points
        if points is not None:
            points = to_numpy(points)
            self.ori_points = points
            self.points = points.astype(np.float32)
            self.check_ori_points_dim()
            if normalize:
                self.normalize_points()
    
        # tours
        if tours is not None:
            tours = to_numpy(tours).astype(np.int32)
            if ref:
                self.ref_tours = tours
                self.check_ref_tours_dim()
            else:
                self.tours = tours
                self.check_tours_dim()

    def to_tsplib_folder(
        self,
        tsp_save_dir: str = None,
        tsp_filename: str = None,
        tour_save_dir: str = None,
        tour_filename: str = None,
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round",
        show_time: bool = False
    ):
        # .tsp files
        if tsp_save_dir is not None:
            # preparation
            if tsp_filename.endswith(".tsp"):
                tsp_filename = tsp_filename.replace(".tsp", "")
            self.check_points_not_none()
            points = self.ori_points if original else self.points
            samples = points.shape[0]

            # apply scale and dtype
            points = self.apply_scale_and_dtype(
                points=points, apply_scale=apply_scale,
                to_int=to_int, round_func=round_func
            )

            # makedirs
            if not os.path.exists(tsp_save_dir):
                os.makedirs(tsp_save_dir)

            # write
            write_msg = f"Writing tsp files to {tsp_save_dir}"
            for idx in iterative_execution(range, samples, write_msg, show_time):
                # file name & save path
                if samples == 1:
                    name = tsp_filename + f".tsp"
                else:
                    name = tsp_filename + f"-{idx}.tsp"
                save_path = os.path.join(tsp_save_dir, name)
                
                # write
                with open(save_path, "w") as f:
                    f.write(f"NAME : {name}\n")
                    f.write(f"COMMENT : Generated by ML4CO-Kit\n")
                    f.write("TYPE : TSP\n")
                    f.write(f"DIMENSION : {self.nodes_num}\n")
                    f.write(f"EDGE_WEIGHT_TYPE : {self.norm}\n")
                    f.write("NODE_COORD_SECTION\n")
                    for i in range(self.nodes_num):
                        x, y = points[idx][i]
                        f.write(f"{i+1} {x} {y}\n")
                    f.write("EOF\n")

        # .opt.tour files
        if tour_save_dir is not None:
            # preparation
            if tour_filename.endswith(".opt.tour"):
                tour_filename = tour_filename.replace(".opt.tour", "")
            if tour_filename.endswith(".tour"):
                tour_filename = tour_filename.replace(".tour", "")
            self.check_tours_not_none(ref=False)
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
        filename: str = "example.txt",
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round"
    ):
        # check
        self.check_points_not_none()
        self.check_tours_not_none(ref=False)
        
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
        points = self.apply_scale_and_dtype(
            points=points, apply_scale=apply_scale,
            to_int=to_int, round_func=round_func
        )

        # write
        with open(filename, "w") as f:
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
        # check
        self.check_points_not_none()
        self.check_tours_not_none(ref=False)
        if calculate_gap:
            self.check_tours_not_none(ref=True)
            
        # variables
        points = self.ori_points if original else self.points
        tours = self.tours
        ref_tours = self.ref_tours

        # apply scale and dtype
        points = self.apply_scale_and_dtype(
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
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "TSPSolver"