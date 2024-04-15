import os
import tsplib95
import numpy as np
from typing import Union
from ml4co_kit.evaluate.tsp.base import TSPEvaluator


SUPPORT_TSPLIB_TYPE = ["EUC_2D", "GEO"]


class TSPSolver:
    def __init__(self, solver_type: str = None, scale: int = 1e6):
        self.solver_type = solver_type
        self.scale = scale
        self.points = None
        self.norm = None
        self.ori_points = None
        self.tours = None
        self.ref_tours = None
        self.nodes_num = None

    def check_points_dim(self):
        self.check_ori_points_dim()
        if self.points is None:
            return
        elif self.points.ndim == 2:
            self.points = np.expand_dims(self.points, axis=0)
        if self.points.ndim != 3:
            raise ValueError("``points`` must be a 2D or 3D array.")
        self.nodes_num = self.points.shape[1]

    def check_ori_points_dim(self):
        if self.ori_points is None:
            return
        elif self.ori_points.ndim == 2:
            self.ori_points = np.expand_dims(self.ori_points, axis=0)
        if self.ori_points.ndim != 3:
            raise ValueError("The ``ori_points`` must be 2D or 3D array.")

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

    def check_points_not_none(self):
        if self.points is None:
            message = (
                "``points`` cannot be None! You can load the points using the methods"
                "``from_data``, ``from_txt``, or ``from_tsp``."
            )
            raise ValueError(message)

    def check_tours_not_none(self):
        if self.tours is None:
            message = (
                "``tours`` cannot be None! You can use solvers based on ``TSPSolver`` "
                "like ``TSPLKHSolver`` or the method ``read_tours`` to get the tours."
            )
            raise ValueError(message)

    def check_ref_tours_not_none(self):
        if self.ref_tours is None:
            message = (
                "``ref_tours`` cannot be None! You can use solvers based on ``TSPSolver`` "
                "like ``TSPLKHSolver`` to obtain the tours and set them as the ``ref_tours``. "
                "Or you can use the methods ``read_ref_tours``, ``from_txt``, "
                "or ``read_ref_tours_from_opt_tour`` to load them."
            )
            raise ValueError(message)

    def set_norm(self, norm: str):
        if norm is None:
            return
        if norm not in SUPPORT_TSPLIB_TYPE:
            message = (
                f"The norm type ({norm}) is not a valid type, "
                f"only {SUPPORT_TSPLIB_TYPE} are supported."
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

    def from_tsp(self, file_path: str, norm: str = "EUC_2D", normalize: bool = False):
        self.set_norm(norm)
        if not file_path.endswith(".tsp"):
            raise ValueError("Invalid file format. Expected a ``.tsp`` file.")
        tsplib_data = tsplib95.load(file_path)
        points = np.array(list(tsplib_data.node_coords.values()))
        if points is None:
            raise RuntimeError("Error in loading {}".format(file_path))
        self.ori_points = points
        self.points = points.astype(np.float32)
        self.check_points_dim()
        if normalize:
            self.normalize_points()

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
            nodes_coords = list()
            ref_tours = list()
            for line in file:
                line = line.strip()
                if "output" in line:
                    split_line = line.split(" output ")
                    points = split_line[0]
                    tour = split_line[1]
                    tour = tour.split(" ")
                    tour = np.array([int(t) for t in tour])
                    tour -= 1
                    ref_tours.append(tour)
                else:
                    points = line
                    load_ref_tours = False
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ]
                )
                nodes_coords.append(points)

        if return_list:
            return nodes_coords, ref_tours

        if load_ref_tours:
            try:
                self.ref_tours = np.array(ref_tours)
            except Exception as e:
                message = (
                    "This method does not support instances of different numbers of nodes. "
                    "If you want to read the data, please set ``return_list`` as True. "
                    "Anyway, the data will not be saved in the solver. "
                    "Please convert the data to ``np.ndarray`` externally before calling the solver."
                )
                raise Exception(message) from e

        nodes_coords = np.array(nodes_coords)
        self.ori_points = nodes_coords
        self.points = nodes_coords.astype(np.float32)
        self.check_points_dim()
        self.check_ref_tours_dim()
        if normalize:
            self.normalize_points()

    def from_data(
        self,
        points: Union[list, np.ndarray],
        norm: str = "EUC_2D",
        normalize: bool = False,
    ):
        self.set_norm(norm)
        if points is None:
            return
        self.ori_points = points
        if type(points) == list:
            points = np.array(points)
        self.points = points.astype(np.float32)
        self.check_points_dim()
        if normalize:
            self.normalize_points()

    def read_tours(self, tours: Union[list, np.ndarray]):
        if tours is None:
            return
        if type(tours) == list:
            tours = np.array(tours)
        self.tours = tours
        self.check_tours_dim()

    def read_ref_tours(self, ref_tours: Union[list, np.ndarray]):
        if ref_tours is None:
            return
        if type(ref_tours) == list:
            ref_tours = np.array(ref_tours)
        self.ref_tours = ref_tours
        self.check_ref_tours_dim()

    def read_ref_tours_from_opt_tour(self, file_path: str):
        if not file_path.endswith(".opt.tour"):
            raise ValueError("Invalid file format. Expected a ``.opt.tour`` file.")
        tsp_tour = tsplib95.load(file_path)
        tsp_tour = tsp_tour.tours
        tsp_tour: list
        tsp_tour = tsp_tour[0]
        tsp_tour.append(1)
        self.ref_tours = np.array(tsp_tour) - 1
        self.check_ref_tours_dim()

    def to_tsp(
        self,
        save_dir: str,
        filename: str,
        original: bool = True,
        points: Union[np.ndarray, list] = None,
        norm: str = None,
        normalize: bool = False,
    ):
        # prepare
        self.from_data(points, norm, normalize)
        if filename.endswith(".tsp"):
            filename = os.path.basename(filename)
        self.check_points_not_none()
        self.check_tours_not_none()
        points = self.ori_points if original else self.points

        # makedirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # write
        for idx in range(points.shape[0]):
            save_path = os.path.join(save_dir, filename + f"-{idx}.tsp")
            with open(save_path, "w") as f:
                f.write(f"NAME: {save_path}\n")
                f.write("TYPE: TSP\n")
                f.write(f"DIMENSION: {self.nodes_num}\n")
                f.write(f"EDGE_WEIGHT_TYPE: {self.norm}\n")
                f.write("NODE_COORD_SECTION\n")
                for i in range(self.nodes_num):
                    x, y = points[idx][i]
                    f.write(f"{i+1} {x} {y}\n")
                f.write("EOF\n")

    def to_opt_tour(
        self,
        save_dir: str,
        filename: str,
        tours: Union[np.ndarray, list] = None,
    ):
        # read and check
        self.read_tours(tours)
        if filename.endswith(".opt.tour"):
            filename = filename.replace(".opt.tour", "")
        self.check_tours_not_none()
        tours = self.tours

        # makedirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # write
        for idx in range(tours.shape[0]):
            save_path = os.path.join(save_dir, filename + f"-{idx}.opt.tour")
            with open(save_path, "w") as f:
                f.write(f"NAME: {save_path}\n")
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
        points: Union[np.ndarray, list] = None,
        tours: Union[np.ndarray, list] = None,
        norm: str = None,
        normalize: bool = False,
    ):
        # read and check
        self.from_data(points, norm, normalize)
        self.read_tours(tours)
        if self.tours is None:
            raise ValueError(
                "``tours`` cannot be None, please use method 'solve' to get solutions."
            )
        self.check_points_not_none()
        self.check_tours_not_none()
        points = self.ori_points if original else self.points
        tours = self.tours

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
        original: bool = True,
        points: Union[np.ndarray, list] = None,
        tours: Union[np.ndarray, list] = None,
        ref_tours: Union[np.ndarray, list] = None,
        norm: str = None,
        normalize: bool = False,
        calculate_gap: bool = False,
    ):
        # read and check
        self.from_data(points, norm, normalize)
        self.read_tours(tours)
        self.read_ref_tours(ref_tours)
        self.check_points_not_none()
        self.check_tours_not_none()
        self.check_ref_tours_not_none()
        points = self.ori_points if original else self.points
        tours = self.tours
        ref_tours = self.ref_tours

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
