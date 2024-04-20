import math
import numpy as np
from typing import Union
from ml4co_kit.evaluate.cvrp.base import CVRPEvaluator
from ml4co_kit.evaluate.tsp.base import geographical


SUPPORT_NORM_TYPE = ["EUC_2D", "GEO"]


class CVRPSolver:
    def __init__(
        self, 
        solver_type: str = None, 
        depots_scale: int = 1e6,
        points_scale: int = 1e6,
        demands_scale: int = 1,
        capacities_scale: int = 1,
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
            )
            raise ValueError(message)
         
    def check_points_not_none(self):
        if self.points is None:
            message = (
            )
            raise ValueError(message)

    def check_demands_not_none(self):
        if self.demands is None:
            message = (
            )
            raise ValueError(message)
    
    def check_tours_not_none(self):
        if self.tours is None:
            message = (
            )
            raise ValueError(message)
        
    def check_ref_tours_not_none(self):
        if self.ref_tours is None:
            message = (
            )
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
            if type(depots) == list:
                depots = np.array(depots)
            self.ori_depots = depots
            self.depots = depots.astype(np.float32)
            self.check_ori_depots_dim()
        
        # points
        if points is not None:
            if type(points) == np.ndarray:
                points = np.array(points)
            self.ori_points = points
            self.points = points.astype(np.float32)
            self.check_ori_points_dim()
        
        # demands
        if demands is not None:
            if type(demands) == list:
                demands = np.array(demands)
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
    
    def read_tours(self, tours: Union[list, np.ndarray]):
        if tours is None:
            return
        if type(tours) == list:
            # 1D tours
            if type(tours[0]) != list:
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
            if type(ref_tours[0]) != list:
                ref_tours = np.array(ref_tours)
            # 2D tours
            else:
                lengths = [len(ref_tours) for ref_tours in ref_tours]
                max_length = max(lengths)
                len_ref_tours = len(ref_tours)
                np_ref_tours = np.zeros(shape=(len_ref_tours, max_length)) - 1
                for idx in range(len_ref_tours):
                    ref_tours = ref_tours[idx]
                    len_ref_tours = len(ref_tours)
                    np_ref_tours[idx][:len_ref_tours] = ref_tours
                ref_tours = np_ref_tours
        self.ref_tours = ref_tours
        self.check_ref_tours_dim()
    
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
                split_demand_need = np.sum(cur_demand[split_tour.astype(int) - 1])
                if split_demand_need > cur_capacity:
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
        check_demands: bool = True
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