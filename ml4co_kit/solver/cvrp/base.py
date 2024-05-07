import os
import sys
import math
import numpy as np
from typing import Union
from pyvrp import Model
from pyvrp import read as read_vrp
from pyvrp import read_solution
from pyvrp.read import ROUND_FUNCS
from ml4co_kit.utils.type import to_numpy
from ml4co_kit.evaluate.cvrp.base import CVRPEvaluator
from ml4co_kit.evaluate.tsp.base import geographical


SUPPORT_NORM_TYPE = ["EUC_2D", "GEO"]
if sys.version_info.major == 3 and sys.version_info.minor == 8:
    CP38 = True
else:
    CP38 = False


class CVRPSolver:
    def __init__(
        self, 
        solver_type: str = None, 
        depots_scale: int = 1e6,
        points_scale: int = 1e6,
        demands_scale: int = 1e6,
        capacities_scale: int = 1e6,
    ):
        self.solver_type = solver_type
        self.depots_scale = depots_scale
        self.points_scale = points_scale
        self.demands_scale = demands_scale
        self.capacities_scale = capacities_scale
        self.depots = None
        self.ori_depots = None
        self.points = None
        self.ori_points = None
        self.demands = None
        self.capacities = None
        self.tours = None
        self.ref_tours = None
        self.nodes_num = None

    def check_depots_dim(self):
        if self.depots is None:
            return
        elif self.depots.ndim == 1:
            self.depots = np.expand_dims(self.depots, axis=0)
        if self.depots.ndim != 2:
            raise ValueError("The dimensions of ``depots`` cannot be larger than 2.")
    
    def check_ori_depots_dim(self):
        self.check_depots_dim()
        if self.ori_depots is None:
            return
        elif self.ori_depots.ndim == 1:
            self.ori_depots = np.expand_dims(self.ori_depots, axis=0)
        if self.ori_depots.ndim != 2:
            raise ValueError("The dimensions of ``ori_depots`` cannot be larger than 2.")
    
    def check_points_dim(self):
        if self.points is None:
            return
        elif self.points.ndim == 2:
            self.points = np.expand_dims(self.points, axis=0)
        if self.points.ndim != 3:
            raise ValueError("``points`` must be a 2D or 3D array.")
        self.nodes_num = self.points.shape[1]

    def check_ori_points_dim(self):
        self.check_points_dim()
        if self.ori_points is None:
            return
        elif self.ori_points.ndim == 2:
            self.ori_points = np.expand_dims(self.ori_points, axis=0)
        if self.ori_points.ndim != 3:
            raise ValueError("The ``ori_points`` must be 2D or 3D array.")
        
    def check_demands_dim(self):
        if self.demands is None:
            return
        elif self.demands.ndim == 1:
            self.demands = np.expand_dims(self.demands, axis=0)
        if self.demands.ndim != 2:
            raise ValueError("The dimensions of ``demands`` cannot be larger than 2.")
    
    def check_capacities_dim(self):
        if self.capacities is None:
            return
        if self.capacities.ndim != 1:
            raise ValueError("The ``capacities`` must be 1D array.")
    
    def check_tours_dim(self):
        if self.tours is None:
            return
        elif self.tours.ndim == 1:
            self.tours = np.expand_dims(self.tours, axis=0)
        if self.tours.ndim != 2:
            raise ValueError("The dimensions of ``tours`` cannot be larger than 2.")

    def check_ref_tours_dim(self):
        if self.ref_tours is None:
            return
        elif self.ref_tours.ndim == 1:
            self.ref_tours = np.expand_dims(self.ref_tours, axis=0)
        if self.ref_tours.ndim != 2:
            raise ValueError(
                "The dimensions of the ``ref_tours`` cannot be larger than 2."
            )

    def check_depots_not_none(self):
        if self.depots is None:
            message = (
                "``depots`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, or ``from_vrp``."
            )
            raise ValueError(message)
         
    def check_points_not_none(self):
        if self.points is None:
            message = (
                "``points`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, or ``from_vrp``."
            )
            raise ValueError(message)

    def check_demands_not_none(self):
        if self.demands is None:
            message = (
                "``demands`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, or ``from_vrp``."
            )
            raise ValueError(message)
    
    def check_capacities_not_none(self):
        if self.demands is None:
            message = (
                "``capacities`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, or ``from_vrp``."
            )
            raise ValueError(message)
    
    def check_tours_not_none(self):
        if self.tours is None:
            message = (
                "``tours`` cannot be None! You can use solvers based on ``CVRPSolver`` "
                "like ``CVRPPyVRPSolver`` or the method ``read_tours`` to get the tours."
            )
            raise ValueError(message)
        
    def check_ref_tours_not_none(self):
        if self.ref_tours is None:
            message = (
                "``ref_tours`` cannot be None! You can use solvers based on ``CVRPSolver`` "
                "like ``CVRPPyVRPSolver`` to obtain the tours and set them as the ``ref_tours``. "
                "Or you can use the methods ``read_ref_tours``, ``from_txt``, "
                "or ``read_ref_tours_from_sol`` to load them."
            )
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
    
    def check_demands_meet(
        self,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
        tours: Union[np.ndarray, list] = None,
    ):
        self.from_data(demands=demands, capacities=capacities)
        self.read_tours(tours)
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

    def get_distance(self, x1: float, x2: float, norm: str = None):
        self.set_norm(norm)
        if self.norm == "EUC_2D":
            return math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
        elif self.norm == "GEO":
            return geographical(x1, x2)
                    
    def from_data(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[int, float, np.ndarray] = None,
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
            if type(capacities) == float or type(capacities) == int:
                capacities = np.array([capacities])
            self.capacities = capacities.astype(np.float32)
            self.capacities = capacities
            self.check_capacities_dim()

        # normalize
        if normalize:
            self.normalize_points_depots()
    
    def from_txt(
        self,
        file_path: str,
        norm: str = "EUC_2D",
        normalize: bool = False,
        load_ref_tours: str = True,
        return_list: bool = False,
    ):
        self.set_norm(norm)

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
            ref_tours = list()
            
            # read by lines
            for line in file:
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
                ref_tour = split_line_4[1]
                
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
                ref_tour = ref_tour.split(" ")
                ref_tour = [int(t) for t in ref_tour]
                
                # add to the list
                depot_list.append(depot)
                points_list.append(points)
                demands_list.append(demands)
                capacity_list.append(capacity)
                ref_tours.append(ref_tour)

        if return_list:
            return depot_list, points_list, demands_list, capacity_list, ref_tours
        
        if load_ref_tours:
            self.read_ref_tours(ref_tours)
        
        depots = np.array(depot_list)
        self.ori_depots = depots
        self.depots = depots.astype(np.float32)
        nodes_coords = np.array(points_list)
        self.ori_points = nodes_coords
        self.points = nodes_coords.astype(np.float32)
        self.demands = np.array(demands_list).astype(np.float32)
        self.capacities = np.array(capacity_list).astype(np.float32)
        self.check_ori_depots_dim()
        self.check_ori_points_dim()
        self.check_ref_tours_dim()
        self.check_demands_dim()
        self.check_capacities_dim()
        if normalize:
            self.normalize_points_depots()
    
    def from_vrp(
        self, 
        file_path: str, 
        round_func: str = "none",
        norm: str = "EUC_2D",
        normalize: bool = False
    ):
        instance = read_vrp(where=file_path, round_func=round_func)
        vrp_model = Model.from_data(instance)
        _depots = vrp_model._depots[0]
        _clients = vrp_model._clients
        _vehicle_types = vrp_model._vehicle_types[0]
        depots = np.array([_depots.x, _depots.y])
        points_list = list()
        demands_list = list()
        for client in _clients:
            points_list.append([client.x, client.y])
            demands_list.append(client.demand if CP38 else client.delivery)
        capacity = _vehicle_types.capacity
        self.from_data(
            depots=depots,
            points=points_list,
            demands=demands_list,
            capacities=capacity,
            norm=norm,
            normalize=normalize
        )        
     
    def read_tours(self, tours: Union[list, np.ndarray]):
        if tours is None:
            return
        if type(tours) == list:
            # 1D tours
            if type(tours[0]) != list and type(tours[0]) != np.ndarray:
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
                tours = np_tours
        self.tours = tours.astype(np.int32)
        self.check_tours_dim()
    
    def read_ref_tours(self, ref_tours: Union[list, np.ndarray]):
        if ref_tours is None:
            return
        if type(ref_tours) == list:
            # 1D tours
            if type(ref_tours[0]) != list and type(ref_tours[0]) != np.ndarray:
                ref_tours = np.array(ref_tours)
            # 2D tours
            else:
                lengths = [len(ref_tours) for ref_tours in ref_tours]
                max_length = max(lengths)
                len_ref_tours = len(ref_tours)
                np_ref_tours = np.zeros(shape=(len_ref_tours, max_length)) - 1
                for idx in range(len_ref_tours):
                    ref_tour = ref_tours[idx]
                    len_ref_tour = len(ref_tour)
                    np_ref_tours[idx][:len_ref_tour] = ref_tour
                ref_tours = np_ref_tours
        self.ref_tours = ref_tours.astype(np.int32)
        self.check_ref_tours_dim()

    def read_ref_tours_from_sol(self, file_path: str):
        if not file_path.endswith(".sol"):
            raise ValueError("Invalid file format. Expected a ``.sol`` file.")
        
        # check the .sol type
        route_flag = None
        with open(file_path, "r") as file:
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
            solution = read_solution(where=file_path)
            tour = [0]
            for _solution in solution:
                tour = tour + _solution + [0]
        elif route_flag == False:
            with open(file_path, "r") as file:
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
            raise ValueError(f"Unable to read route information from {file_path}.")
        
        # read ref tours
        self.read_ref_tours(tour)
        
    def to_txt(
        self,
        filename: str = "example.txt",
        original: bool = True,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[int, float, np.ndarray] = None,
        tours: Union[np.ndarray, list] = None,
        norm: str = None,
        normalize: bool = False,
    ):
        # read and check
        self.from_data(depots, points, demands, capacities, norm, normalize)
        self.read_tours(tours)
        if self.tours is None:
            raise ValueError(
                "``tours`` cannot be None, please use method 'solve' to get solutions."
            )
        self.check_depots_not_none()
        self.check_points_not_none()
        self.check_demands_not_none()
        self.check_capacities_not_none()
        self.check_tours_not_none()
        
        # variables
        _depots = self.ori_depots if original else self.depots
        _points = self.ori_points if original else self.points
        _demands = self.demands
        _capacities = self.capacities
        _tours = self.tours

        # write
        with open(filename, "w") as f:
            # write to txt
            for idx in range(len(_tours)):
                tour = _tours[idx]
                tour = np.split(tour, np.where(tour == -1)[0])[0]
                depot = _depots[idx]
                points = _points[idx]
                demands = _demands[idx]
                capicity = _capacities[idx]
                f.write("depots " + str(depot[0]) + str(" ") + str(depot[1]))
                f.write(" points" + str(" "))
                f.write(
                    " ".join(
                        str(x) + str(" ") + str(y)
                        for x, y in points
                    )
                )
                f.write(" demands " + str(" ").join(str(demand) for demand in demands))
                f.write(" capacity " + str(capicity))
                f.write(str(" output "))
                f.write(str(" ").join(str(node_idx) for node_idx in tour.tolist()))
                f.write("\n")
            f.close()

    def to_vrp(
        self,
        save_dir: str,
        filename: str,
        original: bool = True,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[int, float, np.ndarray] = None,
        norm: str = None,
        normalize: bool = False,
        dtype: str = "int",
        round_func: str = "round"
    ):
        # prepare
        self.from_data(depots, points, demands, capacities, norm, normalize)
        if filename.endswith(".vrp"):
            filename = filename.replace(".vrp", "")
        self.check_depots_not_none()
        self.check_points_not_none()
        self.check_demands_not_none()
        self.check_capacities_not_none()
        
        # variables
        depots = self.ori_depots if original else self.ori_depots
        points = self.ori_points if original else self.points
        demands = self.demands
        capacities = self.capacities
        if dtype == "int":
            self.round_func = self.get_round_func(round_func)
            depots = self.round_func(depots)
            points = self.round_func(points)
            demands = self.round_func(demands)
            capacities = self.round_func(capacities)
            
        # makedirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # write
        for idx in range(points.shape[0]):
            save_path = os.path.join(save_dir, filename + f"-{idx}.vrp")
            with open(save_path, "w") as f:
                f.write(f"NAME : {save_path}\n")
                f.write("TYPE : CVRP\n")
                f.write(f"DIMENSION : {self.nodes_num + 1}\n")
                f.write(f"EDGE_WEIGHT_TYPE : {self.norm}\n")
                f.write(f"CAPACITY : {self.capacities[idx]}\n")
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
    
    def to_sol(
        self,
        save_dir: str,
        filename: str,
        tours: Union[np.ndarray, list] = None,
        dtype: str = "int",
        round_func: str = "round"
    ):
        # read and check
        self.read_tours(tours)
        if filename.endswith(".sol"):
            filename = filename.replace(".sol", "")
        self.check_tours_not_none()
        tours = self.tours
        
        # makedirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # write
        for idx in range(tours.shape[0]):
            save_path = os.path.join(save_dir, filename + f"-{idx}.sol")
            tour = tours[idx]
            split_tours = np.split(tour, np.where(tour == 0)[0])[1: -1]
            with open(save_path, "w") as f:
                for i in range(len(split_tours)):
                    part_tour = split_tours[i][1:]
                    f.write(f"Route #{i+1}: ")
                    f.write(f" ".join(str(int(node)) for node in part_tour))
                    f.write("\n")
                cost = self.evaluate(
                    calculate_gap=False, 
                    dtype=dtype, 
                    round_func=round_func
                )
                f.write(f"Cost {cost}\n")
    
    def modify_tour(self, tour: np.ndarray):
        if not np.isin(-1, tour):
            return tour
        return tour[: np.where(tour == -1)[0][0]]
    
    def evaluate(
        self,
        original: bool = True,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        demands: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
        norm: str = None,
        normalize: bool = False,
        tours: Union[np.ndarray, list] = None,
        ref_tours: Union[np.ndarray, list] = None,
        calculate_gap: bool = False,
        check_demands: bool = True,
        dtype: str = "float",
        round_func: str = "none"
    ):
        # read and check
        self.from_data(depots, points, demands, capacities, norm, normalize)
        self.read_tours(tours)
        self.read_ref_tours(ref_tours)
        self.check_points_not_none()
        self.check_tours_not_none()
        if check_demands:
            self.check_demands_meet()
        if calculate_gap:
            self.check_ref_tours_not_none()
        depots = self.ori_depots if original else self.depots
        points = self.ori_points if original else self.points
        tours = self.tours
        ref_tours = self.ref_tours

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
            solved_cost = evaluator.evaluate(self.modify_tour(solved_tour), dtype, round_func)
            tours_cost_list.append(solved_cost)
            if calculate_gap:
                ref_cost = evaluator.evaluate(self.modify_tour(ref_tours[idx]), dtype, round_func)
                ref_tours_cost_list.append(ref_cost)
                gap = (solved_cost - ref_cost) / ref_cost * 100
                gap_list.append(gap)

        # calculate average cost/gap & std
        tours_costs = np.array(tours_cost_list)
        if calculate_gap:
            ref_costs = np.array(ref_tours_cost_list)
            gaps = np.array(gap_list)
        costs_avg = np.average(tours_costs)
        if dtype == "int":
            costs_avg = int(costs_avg)
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
    