import os
import sys
import math
import numpy as np
from typing import Union
from pyvrp import Model
from pyvrp import read as read_vrp
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.evaluate.cvrp.base import CVRPEvaluator
from ml4co_kit.utils.distance_utils import geographical
from ml4co_kit.utils.type_utils import to_numpy, TASK_TYPE, SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, iterative_execution_for_file


SUPPORT_NORM_TYPE = ["EUC_2D", "GEO"]
if sys.version_info.major == 3 and sys.version_info.minor == 8:
    CP38 = True
    from pyvrp.read import ROUND_FUNCS
else:
    CP38 = False
    from ml4co_kit.utils.round import ROUND_FUNCS


class CVRPSolver(SolverBase):
    def __init__(
        self, 
        solver_type: str = None, 
        depots_scale: int = 1e4,
        points_scale: int = 1e4,
        demands_scale: int = 1e3,
        capacities_scale: int = 1e3,
    ):
        super(CVRPSolver, self).__init__(
            task_type=TASK_TYPE.CVRP, solver_type=solver_type
        )
        self.solver_type = solver_type
        self.depots_scale = depots_scale
        self.points_scale = points_scale
        self.demands_scale = demands_scale
        self.capacities_scale = capacities_scale
        self.depots: np.ndarray = None
        self.ori_depots: np.ndarray = None
        self.points: np.ndarray = None
        self.ori_points: np.ndarray = None
        self.demands: np.ndarray = None
        self.capacities: np.ndarray = None
        self.tours: np.ndarray = None
        self.ref_tours: np.ndarray = None
        self.nodes_num: int = None
        self.norm: str = None
        
    def check_depots_dim(self):
        if self.depots is not None:
            if self.depots.ndim == 1:
                self.depots = np.expand_dims(self.depots, axis=0)
            if self.depots.ndim != 2:
                raise ValueError("The dimensions of ``depots`` cannot be larger than 2.")
        
    def check_ori_depots_dim(self):
        self.check_depots_dim()
        if self.ori_depots is not None:
            if self.ori_depots.ndim == 1:
                self.ori_depots = np.expand_dims(self.ori_depots, axis=0)
            if self.ori_depots.ndim != 2:
                raise ValueError("The dimensions of ``ori_depots`` cannot be larger than 2.")
    
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
            
    def check_demands_dim(self):
        if self.demands is not None:
            if self.demands.ndim == 1:
                self.demands = np.expand_dims(self.demands, axis=0)
            if self.demands.ndim != 2:
                raise ValueError("The dimensions of ``demands`` cannot be larger than 2.")
    
    def check_capacities_dim(self):
        if self.capacities is not None:
            if self.capacities.ndim != 1:
                raise ValueError("The ``capacities`` must be 1D array.")
    
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

    def check_depots_not_none(self):
        if self.depots is None:
            message = (
                "``depots`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, ``from_vrplib``, or ``from_vrplib_folder``."
            )
            raise ValueError(message)
         
    def check_points_not_none(self):
        if self.points is None:
            message = (
                "``points`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, ``from_vrplib``, or ``from_vrplib_folder``."
            )
            raise ValueError(message)

    def check_demands_not_none(self):
        if self.demands is None:
            message = (
                "``demands`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, ``from_vrplib``, or ``from_vrplib_folder``."
            )
            raise ValueError(message)
    
    def check_capacities_not_none(self):
        if self.demands is None:
            message = (
                "``capacities`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, ``from_vrplib``, or ``from_vrplib_folder``."
            )
            raise ValueError(message)
    
    def check_tours_not_none(self, ref: bool):
        msg = "ref_tours" if ref else "tours"
        message = (
            f"``{msg}`` cannot be None! You can use solvers based on "
            "``CVRPSolver`` like ``CVRPHGSSolver`` or use methods including "
            "``from_data``, ``from_txt`` or ``from_vrplib`` to obtain them."
        )  
        if ref:
            if self.ref_tours is None:
                raise ValueError(message)
        else:
            if self.tours is None:    
                raise ValueError(message)
    
    def get_round_func(self, round_func: str):
        if (key := str(round_func)) in ROUND_FUNCS:
            round_func = ROUND_FUNCS[key]
        if not callable(round_func):
            raise TypeError(
                f"round_func = {round_func} is not understood. Can be a function,"
                f" or one of {ROUND_FUNCS.keys()}."
            )
        return round_func
    
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

    def normalize_points_depots(self):
        for idx in range(self.points.shape[0]):
            cur_points = self.points[idx]
            cur_depots = self.depots[idx]
            max_value = np.max(cur_points)
            min_value = np.min(cur_points)
            cur_points = (cur_points - min_value) / (max_value - min_value)
            cur_depots = (cur_depots - min_value) / (max_value - min_value)
            self.points[idx] = cur_points
            self.depots[idx] = cur_depots
    
    def check_demands_meet(self):
        tours_shape = self.tours.shape
        for idx in range(tours_shape[0]):
            cur_demand = self.demands[idx]
            cur_capacity = self.capacities[idx]
            cur_tour = self.tours[idx]
            split_tours = np.split(cur_tour, np.where(cur_tour == 0)[0])[1: -1]
            for split_idx in range(len(split_tours)):
                split_tour = split_tours[split_idx][1:]
                split_demand_need = np.sum(cur_demand[split_tour.astype(int) - 1], dtype=np.float32)
                if split_demand_need > cur_capacity + 1e-5:
                    message = (
                        f"Capacity constraint not met in tour {idx}. "
                        f"The split tour is ``{split_tour}`` with the demand of {split_demand_need}."
                        f"However, the maximum capacity of the vehicle is {cur_capacity}."
                    )
                    raise ValueError(message)
    
    def modify_tour(self, tour: np.ndarray):
        if not np.isin(-1, tour):
            return tour
        return tour[: np.where(tour == -1)[0][0]]

    def get_distance(self, x1: float, x2: float, norm: str = None):
        self.set_norm(norm)
        if self.norm == "EUC_2D":
            return math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
        elif self.norm == "GEO":
            return geographical(x1, x2)

    def apply_scale_and_dtype(
        self, 
        depots: np.ndarray,
        points: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        apply_scale: bool,
        to_int: bool,
        round_func: str
    ):
        # apply scale
        if apply_scale:
            depots = depots * self.depots_scale
            points = points * self.points_scale
            demands = demands * self.demands_scale
            capacities = capacities * self.capacities_scale
        
        # dtype
        if to_int:
            round_func = self.get_round_func(round_func)
            depots = round_func(depots)
            points = round_func(points)
            demands = round_func(demands)
            capacities = round_func(capacities)
        
        return depots, points, demands, capacities
    
    def _read_data_from_vrp_file(self, vrp_file_path: str, round_func: str):
        # instance and model
        instance = read_vrp(where=vrp_file_path, round_func=round_func)
        vrp_model = Model.from_data(instance)
        
        # depots
        _depots = vrp_model._depots[0]
        depots = np.array([_depots.x, _depots.y])
        
        # points and demands
        _clients = vrp_model._clients
        _vehicle_types = vrp_model._vehicle_types[0]
        points_list = list()
        demands_list = list()
        for client in _clients:
            points_list.append([client.x, client.y])
            demands_list.append(client.demand if CP38 else client.delivery)
        points = np.array(points_list)
        demands = np.array(demands_list).reshape(-1)
        
        # capacity
        capacity = _vehicle_types.capacity
        
        # return
        return depots, points, demands, capacity

    def _read_tour_from_sol_file(self, sol_file_path: str = None):
        # check the .sol type
        route_flag = None
        with open(sol_file_path, "r") as file:
            first_line = file.readline()
            if "Route" in first_line:
                # Like this
                # Route #1: 15 17 9 3 16 29
                # Route #2: 12 5 26 7 8 13 32 2
                route_flag = True
            else:
                # Like this
                # 36395
                # 37
                # 1893900
                # 1133620
                # 0 1 1 1144 12  14 0 217 236 105 2 169 8 311 434 362 187 136 59 0
                # 0 1 1 1182 12  14 0 3 370 133 425 349 223 299 386 267 410 411 348 0
                route_flag = False
        
        # read the data form .sol
        if route_flag == True:
            with open(sol_file_path, "r") as file:
                tour = [0]
                for line in file:
                    if line.startswith("Route"):
                        split_line = line.replace("\n", "").split(":")[1][1:].split(" ")
                        for node in split_line:
                            tour.append(int(node))
                        tour.append(0)
        elif route_flag == False:
            with open(sol_file_path, "r") as file:
                line_idx = 0
                tour = [0]
                for line in file:
                    line_idx += 1
                    if line_idx < 5:
                        continue
                    split_line = line.split(" ")[7:-1]
                    for node in split_line:
                        tour.append(int(node))
                    tour.append(0)
        else:
            raise ValueError(
                f"Unable to read route information from {sol_file_path}."
            )
            
        return tour

    def from_vrplib(
        self, 
        vrp_file_path: str = None, 
        sol_file_path: str = None, 
        round_func: str = "none",
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False
    ):
        # init
        depots = None
        points = None
        demands = None
        capacity = None
        tour = None
        
        # read problem from .vrp file
        if vrp_file_path is not None:
            if not vrp_file_path.endswith(".vrp"):
                raise ValueError("Invalid file format. Expected a ``.vrp`` file.")
            depots, points, demands, capacity = \
                self._read_data_from_vrp_file(vrp_file_path, round_func)
            
        # read solution from .sol file
        if sol_file_path is not None:
            if not sol_file_path.endswith(".sol"):
                raise ValueError("Invalid file format. Expected a ``.sol`` file.")
            tour = self._read_tour_from_sol_file(sol_file_path)
            
        # use ``from_data``
        self.from_data(
            depots=depots, points=points, demands=demands, capacities=capacity, 
            tours=tour, ref=ref, norm=norm, normalize=normalize
        )
        
    def from_vrplib_folder(
        self, 
        vrp_folder_path: str = None,
        sol_folder_path: str = None,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time: bool = False
    ):
        # init
        depots = None
        points = None
        demands = None
        capacities = None
        tours = None
        vrp_flag = False if vrp_folder_path is None else True
        sol_flag = False if sol_folder_path is None else True
        
        # only data
        if vrp_flag and not sol_flag:
            depots_list = list()
            points_list = list()
            demands_list = list()
            capacity_list = list()
            files = os.listdir(vrp_folder_path)
            files.sort()
            for file_name in iterative_execution_for_file(files, "Loading", show_time):
                vrp_file_path = os.path.join(vrp_folder_path, file_name)
                if not vrp_file_path.endswith(".vrp"):
                    continue
                depots, points, demands, capacity = \
                    self._read_data_from_vrp_file(vrp_file_path)
                depots_list.append(depots)
                points_list.append(points)
                demands_list.append(demands)
                capacity_list.append(capacity)

        # only sol
        if not vrp_flag and sol_flag:
            tours_list = list()
            files = os.listdir(sol_folder_path)
            files.sort()
            for file_name in iterative_execution_for_file(files, "Loading", show_time):
                sol_file_path = os.path.join(sol_folder_path, file_name)
                if not sol_file_path[-4:] == ".sol":
                    continue
                tour = self._read_tour_from_sol_file(sol_file_path)
                tours_list.append(tour)
        
        # both points and tours [must have the same filename]
        if vrp_flag and sol_flag:
            depots_list = list()
            points_list = list()
            demands_list = list()
            capacity_list = list()
            tours_list = list()
            files = os.listdir(vrp_folder_path)
            files.sort()
            for file_name in iterative_execution_for_file(files, "Loading", show_time):
                # data
                vrp_file_path = os.path.join(vrp_folder_path, file_name)
                if not sol_file_path.endswith(".vrp"):
                    continue
                depots, points, demands, capacity = \
                    self._read_data_from_vrp_file(sol_file_path)
                depots_list.append(depots)
                points_list.append(points)
                demands_list.append(demands)
                capacity_list.append(capacity)
                # sol
                sol_file_path = os.path.join(
                    sol_folder_path, file_name.replace(".vrp", ".sol")
                )
                tour = self._read_tour_from_sol_file(sol_file_path)
                tours_list.append(tour)
        
        # return list
        if return_list:
            if vrp_flag:
                if sol_flag:
                    return depots_list, points_list, demands_list, capacity_list, tours_list
                else:
                    return depots_list, points_list, demands_list, capacity_list
            else:
                if sol_flag:
                    return tours_list
        
        # check
        message = (
            "This method does not support instances of different numbers of nodes. "
            "If you want to read the data, please set ``return_list`` as True. "
            "Anyway, the data will not be saved in the solver. "
            "Please convert the data to ``np.ndarray`` externally before calling the solver."
        )
        if vrp_flag:
            try:
                depots = np.array(depots_list)
                points = np.array(points_list)
                demands = np.array(demands_list)
                capacities = np.array(capacity_list)
            except Exception as e:
                raise Exception(message) from e
        if sol_flag:
            try:
                tours = np.array(tours_list)
            except Exception as e:
                raise Exception(message) from e
        
        # use ``from_data``
        self.from_data(
            depots=depots, points=points, demands=demands, capacities=capacities,
            tours=tours, ref=ref, norm=norm, normalize=normalize
        )        

    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        return_list: bool = False,
        show_time: bool = False
    ):
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            # record to lists
            depot_list = list()
            points_list = list()
            demands_list = list()
            capacity_list = list()
            tours = list()
            
            # read by lines
            for line in iterative_execution_for_file(file, "Loading", show_time):
                # line to strings
                line = line.strip()
                split_line_0 = line.split("depots ")[1]
                split_line_1 = split_line_0.split(" points ")
                depot = split_line_1[0]
                split_line_2 = split_line_1[1].split(" demands ")
                points = split_line_2[0]
                split_line_3 = split_line_2[1].split(" capacity ")
                demands = split_line_3[0]
                split_line_4 = split_line_3[1].split(" output ")
                capacity = split_line_4[0]
                tour = split_line_4[1]
                
                # strings to array
                depot = depot.split(" ")
                depot = np.array([float(depot[0]), float(depot[1])])
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ]
                )
                demands = demands.split(" ")
                demands = np.array([
                    float(demands[i]) for i in range(len(demands))
                ])
                capacity = float(capacity)
                tour = tour.split(" ")
                tour = [int(t) for t in tour]
                
                # add to the list
                depot_list.append(depot)
                points_list.append(points)
                demands_list.append(demands)
                capacity_list.append(capacity)
                tours.append(tour)

        # check if return list
        if return_list:
            return depot_list, points_list, demands_list, capacity_list, tours
        
        # use ``from_data``
        self.from_data(
            depots=depot_list, points=points_list, 
            demands=demands_list, capacities=capacity_list, 
            tours=tours, ref=ref, norm=norm, normalize=normalize
        )
              
    def from_data(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[int, float, np.ndarray] = None,
        tours: Union[list, np.ndarray] = None,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False
    ):
        # norm
        self.set_norm(norm)
        
        # depots
        if depots is not None:
            depots = to_numpy(depots)
            self.ori_depots = depots
            self.depots = depots.astype(np.float32)
            self.check_ori_depots_dim()
        
        # points
        if points is not None:
            points = to_numpy(points)
            self.ori_points = points
            self.points = points.astype(np.float32)
            self.check_ori_points_dim()
        
        # demands
        if demands is not None:
            demands = to_numpy(demands)
            self.demands = demands.astype(np.float32)
            self.check_demands_dim()
        
        # capacities
        if capacities is not None:
            if isinstance(capacities, (float, int)):
                capacities = np.array([capacities])
            if isinstance(capacities, list):
                capacities = np.array(capacities)
            self.capacities = capacities.astype(np.float32)
            self.capacities = capacities
            self.check_capacities_dim()

        # tours (ref or not)
        if tours is not None:
            if isinstance(tours, list):
                # 1D tours
                if not isinstance(tours[0], list) and not isinstance(tours[0], np.ndarray):
                    tours = np.array(tours)
                # 2D tours
                else:
                    lengths = [len(tour) for tour in tours]
                    max_length = max(lengths)
                    len_tours = len(tours)
                    np_tours = np.zeros(shape=(len_tours, max_length)) - 1
                    for idx in range(len_tours):
                        tour = tours[idx]
                        len_tour = len(tour)
                        np_tours[idx][:len_tour] = tour
                    tours = np_tours.astype(np.int32)
            if ref:
                self.ref_tours = tours
                self.check_ref_tours_dim()
            else:
                self.tours = tours
                self.check_tours_dim()
                
        # normalize
        if normalize:
            self.normalize_points_depots()

    def to_vrplib_folder(
        self,
        vrp_save_dir: str = None,
        vrp_filename: str = None,
        sol_save_dir: str = None,
        sol_filename: str = None,
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round",
        show_time: bool = False
    ):
        # check
        self.check_depots_not_none()
        self.check_points_not_none()
        self.check_demands_not_none()
        self.check_capacities_not_none()

        # variables
        depots = self.ori_depots if original else self.ori_depots
        points = self.ori_points if original else self.points
        demands = self.demands
        capacities = self.capacities
        samples = points.shape[0]

        # apply scale and dtype
        depots, points, demands, capacities = self.apply_scale_and_dtype(
            depots=depots, points=points, demands=demands, capacities=capacities, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )
        
        # demands and capacities need be int
        demands = demands.astype(np.int32)
        capacities = capacities.astype(np.int32)

        # .vrp files
        if vrp_save_dir is not None:
            # filename
            if vrp_filename.endswith(".vrp"):
                vrp_filename = vrp_filename.replace(".vrp", "")

            # makedirs
            if not os.path.exists(vrp_save_dir):
                os.makedirs(vrp_save_dir)

            # write
            write_msg = f"Writing vrp files to {vrp_save_dir}"
            for idx in iterative_execution(range, samples, write_msg, show_time):
                # file name & save path
                if samples == 1:
                    name = vrp_filename + f".vrp"
                else:
                    name = vrp_filename + f"-{idx}.vrp"
                save_path = os.path.join(vrp_save_dir, name)
                
                # write
                with open(save_path, "w") as f:
                    f.write(f"NAME : {name}\n")
                    f.write(f"COMMENT : Generated by ML4CO-Kit\n")
                    f.write("TYPE : CVRP\n")
                    f.write(f"DIMENSION : {self.nodes_num + 1}\n")
                    f.write(f"EDGE_WEIGHT_TYPE : {self.norm}\n")
                    f.write(f"CAPACITY : {capacities[idx]}\n")
                    f.write("NODE_COORD_SECTION\n")
                    x, y = depots[idx]
                    f.write(f"1 {x} {y}\n")
                    for i in range(self.nodes_num):
                        x, y = points[idx][i]
                        f.write(f"{i+2} {x} {y}\n")
                    f.write("DEMAND_SECTION \n")
                    f.write(f"1 0\n")
                    for i in range(self.nodes_num):
                        f.write(f"{i+2} {demands[idx][i]}\n")
                    f.write("DEPOT_SECTION \n")
                    f.write("	1\n")
                    f.write("	-1\n")
                    f.write("EOF\n")
        
        # .sol files
        if sol_save_dir is not None:
            # check
            self.check_tours_not_none(ref=False)
            
            # variables
            tours = self.tours   
            
            # filename
            if sol_filename.endswith(".sol"):
                sol_filename = sol_filename.replace(".sol", "")

            # makedirs
            if not os.path.exists(sol_save_dir):
                os.makedirs(sol_save_dir)

            # write
            write_msg = f"Writing soution files to {sol_save_dir}"
            for idx in iterative_execution(range, samples, write_msg, show_time):
                # file name & save path
                if samples == 1:
                    name = sol_filename + f".sol"
                else:
                    name = sol_filename + f"-{idx}.sol"
                save_path = os.path.join(sol_save_dir, name)

                # evaluator
                evaluator = CVRPEvaluator(depots[idx], points[idx], self.norm)

                # write
                tour = tours[idx]
                split_tours = np.split(tour, np.where(tour == 0)[0])[1: -1]
                with open(save_path, "w") as f:
                    for i in range(len(split_tours)):
                        part_tour = split_tours[i][1:]
                        f.write(f"Route #{i+1}: ")
                        f.write(f" ".join(str(int(node)) for node in part_tour))
                        f.write("\n")
                        
                    cost = evaluator.evaluate(
                        route=self.modify_tour(tour), 
                        to_int=to_int, 
                        round_func=round_func
                    )
                    f.write(f"Cost {cost}\n")
                
    def to_txt(
        self,
        filename: str = "example.txt",
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round"
    ):
        # check
        self.check_depots_not_none()
        self.check_points_not_none()
        self.check_demands_not_none()
        self.check_capacities_not_none()
        self.check_tours_not_none(ref=False)
        
        # variables
        depots = self.ori_depots if original else self.depots
        points = self.ori_points if original else self.points
        demands = self.demands
        capacities = self.capacities
        tours = self.tours
        
        # apply scale and dtype
        depots, points, demands, capacities = self.apply_scale_and_dtype(
            depots=depots, points=points, demands=demands, capacities=capacities, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )
        
        # write
        with open(filename, "w") as f:
            # write to txt
            for idx in range(len(tours)):
                tour = tours[idx]
                _tour = np.split(tour, np.where(tour == -1)[0])[0]
                _depot = depots[idx]
                _points = points[idx]
                _demands = demands[idx]
                _capicity = capacities[idx]
                f.write("depots " + str(_depot[0]) + str(" ") + str(_depot[1]))
                f.write(" points" + str(" "))
                f.write(
                    " ".join(
                        str(x) + str(" ") + str(y)
                        for x, y in _points
                    )
                )
                f.write(" demands " + str(" ").join(str(demand) for demand in _demands))
                f.write(" capacity " + str(_capicity))
                f.write(str(" output "))
                f.write(str(" ").join(str(node_idx) for node_idx in _tour.tolist()))
                f.write("\n")
            f.close()

    
    def evaluate(
        self,
        calculate_gap: bool = False,
        check_demands: bool = True,
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round",
    ):
        # check
        self.check_points_not_none()
        self.check_tours_not_none(ref=False)
        if check_demands:
            self.check_demands_meet()
        if calculate_gap:
            self.check_tours_not_none(ref=False)
            
        # variables
        depots = self.ori_depots if original else self.depots
        points = self.ori_points if original else self.points
        demands = self.demands
        capacities = self.capacities
        tours = self.tours
        ref_tours = self.ref_tours

        # apply scale and dtype
        depots, points, demands, capacities = self.apply_scale_and_dtype(
            depots=depots, points=points, demands=demands, capacities=capacities, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )
        
        # prepare for evaluate
        tours_cost_list = list()
        samples = points.shape[0]
        if calculate_gap:
            ref_tours_cost_list = list()
            gap_list = list()
            
        # evaluate
        for idx in range(samples):
            evaluator = CVRPEvaluator(depots[idx], points[idx], self.norm)
            solved_tour = tours[idx]
            solved_cost = evaluator.evaluate(
                route=self.modify_tour(solved_tour), 
                to_int=to_int, 
                round_func=round_func
            )
            tours_cost_list.append(solved_cost)
            if calculate_gap:
                ref_cost = evaluator.evaluate(
                    route=self.modify_tour(ref_tours[idx]), 
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
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
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
        return "CVRPSolver"