import os
import math
import numpy as np
from typing import Union
from ml4co_kit.utils import tsplib95
from ml4co_kit.utils.type import to_numpy
from ml4co_kit.evaluate.atsp.base import ATSPEvaluator
from ml4co_kit.utils.run_utils import iterative_execution, iterative_execution_for_file


class ATSPSolver:
    def __init__(self, solver_type: str = None, scale: int = 1e6):
        self.solver_type = solver_type
        self.scale = scale
        self.dists = None
        self.ori_dists = None
        self.tours = None
        self.ref_tours = None
        self.nodes_num = None

    def check_dists_dim(self):
        if self.dists is None:
            return
        elif self.dists.ndim == 2:
            self.dists = np.expand_dims(self.dists, axis=0)
        if self.dists.ndim != 3:
            raise ValueError("``dists`` must be a 2D or 3D array.")
        self.nodes_num = self.dists.shape[-1]

    def check_ori_dists_dim(self):
        self.check_dists_dim()
        if self.ori_dists is None:
            return
        elif self.ori_dists.ndim == 2:
            self.ori_dists = np.expand_dims(self.ori_dists, axis=0)
        if self.ori_dists.ndim != 3:
            raise ValueError("The ``ori_dists`` must be 2D or 3D array.")

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

    def check_dists_not_none(self):
        if self.dists is None:
            message = (
                "``dists`` cannot be None! You can load the dists using the methods"
                "``from_data``, ``from_txt``, ``from_atsp`` or ``from_atsp_folder``."
            )
            raise ValueError(message)

    def check_tours_not_none(self):
        if self.tours is None:
            message = (
                "``tours`` cannot be None! You can use solvers based on ``ATSPSolver`` "
                "like ``ATSPLKHSolver`` or the method ``read_tours`` to get the tours."
            )
            raise ValueError(message)

    def check_ref_tours_not_none(self):
        if self.ref_tours is None:
            message = (
                "``ref_tours`` cannot be None! You can use solvers based on ``ATSPSolver`` "
                "like ``ATSPLKHSolver`` to obtain the tours and set them as the ``ref_tours``. "
                "Or you can use the methods ``read_ref_tours``, ``from_txt``, ``read_ref_tours_from_folder``, "
                "or ``read_ref_tours_from_opt_tour`` to load them."
            )
            raise ValueError(message)

    def from_atsp(self, file_path: str):
        if not file_path.endswith(".atsp"):
            raise ValueError("Invalid file format. Expected a ``.atsp`` file.")
        tsplib_data = tsplib95.load(file_path)
        dists = np.array(tsplib_data.edge_weights)
        if dists is None:
            raise RuntimeError("Error in loading {}".format(file_path))
        self.ori_dists = dists
        self.dists = dists.astype(np.float32)
        self.check_ori_dists_dim()

    def from_atsp_folder(
        self, 
        folder_path: str, 
        return_list: bool = False, 
        show_time: bool = False
    ):
        dists = list()
        files = os.listdir(folder_path)
        for file_name in iterative_execution_for_file(files, "Loading", show_time):
            file_path = os.path.join(folder_path, file_name)
            if not file_path.endswith(".atsp"):
                continue
            tsplib_data = tsplib95.load(file_path)
            dist = np.array(tsplib_data.edge_weights)
            if dist is None:
                raise RuntimeError("Error in loading {}".format(file_path))
            dists.append(dist)
        
        if return_list:
            return dists
        else:
            try:
                dists = np.array(dists)
            except Exception as e:
                message = (
                    "This method does not support instances of different numbers of nodes. "
                    "If you want to read the data, please set ``return_list`` as True. "
                    "Anyway, the data will not be saved in the solver. "
                    "Please convert the data to ``np.ndarray`` externally before calling the solver."
                )
                raise Exception(message) from e 
            self.ori_dists = dists     
            self.dists = dists.astype(np.float32)
            self.check_ori_dists_dim()

    def from_txt(
        self, 
        file_path: str, 
        load_ref_tours: str = True, 
        return_list: bool = False,
        show_time: bool = False
    ):
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            dists = list()
            ref_tours = list()
            for line in iterative_execution_for_file(file, "Loading", show_time):
                line = line.strip()
                if "output" in line:
                    split_line = line.split(" output ")
                    dist = split_line[0]
                    tour = split_line[1]
                    tour = tour.split(" ")
                    tour = np.array([int(t) for t in tour])
                    tour -= 1
                    ref_tours.append(tour)
                else:
                    dist = line
                    load_ref_tours = False
                dist = dist.split(" ")
                dist.append('')           
                dist = np.array([float(dist[2*i]) for i in range(len(dist) // 2)])
                num_nodes = int(math.sqrt(len(dist)))
                dist = dist.reshape(num_nodes, num_nodes)
                dists.append(dist)

        if return_list:
            return dists, ref_tours

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

        dists = np.array(dists)
        self.ori_dists = dists
        self.dists = dists.astype(np.float32)
        self.check_ori_dists_dim()
        self.check_ref_tours_dim()

    def from_data(self, dists: Union[list, np.ndarray]):
        if dists is None:
            return
        self.ori_dists = dists
        dists = to_numpy(dists)
        self.ori_dists = dists
        self.dists = dists.astype(np.float32)
        self.check_ori_dists_dim()

    def read_tours(self, tours: Union[list, np.ndarray]):
        if tours is None:
            return
        tours = to_numpy(tours)
        self.tours = tours.astype(np.int32)
        self.check_tours_dim()

    def read_ref_tours(self, ref_tours: Union[list, np.ndarray]):
        if ref_tours is None:
            return
        ref_tours = to_numpy(ref_tours)
        self.ref_tours = ref_tours.astype(np.int32)
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

    def read_ref_tours_from_folder(
        self, 
        folder_path: str, 
        return_list: bool = False, 
        show_time: bool = False
    ):
        ref_tours = list()
        files = os.listdir(folder_path)
        for file_name in iterative_execution_for_file(files, "Loading", show_time):
            file_path = os.path.join(folder_path, file_name)
            if not file_path.endswith(".opt.tour"):
                continue
            tsp_tour = tsplib95.load(file_path)
            tsp_tour = tsp_tour.tours
            tsp_tour: list
            tsp_tour = tsp_tour[0]
            tsp_tour.append(1)
            ref_tour = np.array(tsp_tour) - 1
            ref_tours.append(ref_tour)
        
        if return_list:
            return ref_tours
        else:
            try:
                ref_tours = np.array(ref_tours)
            except Exception as e:
                message = (
                    "This method does not support instances of different numbers of nodes. "
                    "If you want to read the data, please set ``return_list`` as True. "
                    "Anyway, the data will not be saved in the solver. "
                    "Please convert the data to ``np.ndarray`` externally before calling the solver."
                )
                raise Exception(message) from e 
            self.ref_tours = ref_tours.astype(np.int32)     
            self.check_ref_tours_dim()

    def to_atsp_folder(
        self,
        save_dir: str,
        filename: str,
        original: bool = True,
        dists: Union[np.ndarray, list] = None,
        show_time: bool = False
    ):
        # prepare
        self.from_data(dists)
        if filename.endswith(".atsp"):
            filename = filename.replace(".atsp", "")
        self.check_dists_not_none()
        dists = self.ori_dists if original else self.dists

        # makedirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # write
        for idx in iterative_execution(range, dists.shape[0], "Writing", show_time):
            save_path = os.path.join(save_dir, filename + f"-{idx}.atsp")
            with open(save_path, "w") as f:
                f.write(f"NAME : Generated by ML4CO-Kit\n")
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

    def to_opt_tour_folder(
        self,
        save_dir: str,
        filename: str,
        tours: Union[np.ndarray, list] = None,
        show_time: bool = False
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
        for idx in iterative_execution(range, tours.shape[0], "Writing", show_time):
            save_path = os.path.join(save_dir, filename + f"-{idx}.opt.tour")
            with open(save_path, "w") as f:
                f.write(f"NAME: {save_path}\n")
                f.write(f"TYPE: TOUR\n")
                f.write(f"DIMENSION: {self.nodes_num}\n")
                f.write(f"TOUR_SECTION\n")
                for i in range(self.nodes_num):
                    f.write(f"{tours[idx][i] + 1}\n")
                f.write(f"-1\n")
                f.write(f"EOF\n")

    def to_txt(
        self,
        filename: str = "example.txt",
        original: bool = True,
        dists: Union[np.ndarray, list] = None,
        tours: Union[np.ndarray, list] = None,
    ):
        # read and check
        self.from_data(dists)
        self.read_tours(tours)
        if self.tours is None:
            raise ValueError(
                "``tours`` cannot be None, please use method 'solve' to get solutions."
            )
        self.check_dists_not_none()
        self.check_tours_not_none()
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

        # write
        with open(filename, "w") as f:
            for dist, tour in zip(dists, tours):
                dist: np.ndarray = dist.reshape(-1)
                f.write(" ".join(str(x) + str(" ") for x in dist))
                f.write(str("output") + str(" "))
                f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                f.write("\n")
            f.close()

    def evaluate(
        self,
        original: bool = True,
        dists: Union[np.ndarray, list] = None,
        tours: Union[np.ndarray, list] = None,
        ref_tours: Union[np.ndarray, list] = None,
        calculate_gap: bool = False,
    ):
        # read and check
        self.from_data(dists)
        self.read_tours(tours)
        self.read_ref_tours(ref_tours)
        self.check_dists_not_none()
        self.check_tours_not_none()
        if calculate_gap:
            self.check_ref_tours_not_none()
        dists = self.ori_dists if original else self.dists
        tours = self.tours
        ref_tours = self.ref_tours

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
                evaluator = ATSPEvaluator(dists[idx])
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
        dists: Union[np.ndarray, list] = None,
        num_threads: int = 1,
        show_time: bool = False,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )
