import os
import sys
import time
import shutil
import numpy as np
import pathlib
from tqdm import tqdm
from typing import Union, Sequence
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.solver import (
    CVRPSolver, CVRPPyVRPSolver, CVRPLKHSolver, CVRPHGSSolver
)


class CVRPDataGenerator:
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num: int = 50,
        data_type: str = "uniform",
        solver: Union[SOLVER_TYPE, CVRPSolver] = SOLVER_TYPE.HGS,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/cvrp/uniform",
        filename: str = None,
        # special for demand and capacity
        min_demand: int = 1,
        max_demand: int = 10,
        min_capacity: int = 40,
        max_capacity: int = 40,
        # special for gaussian
        gaussian_mean_x: float = 0.0,
        gaussian_mean_y: float = 0.0,
        gaussian_std: float = 1.0,
    ):
        """
        CVRPDataGenerator
        Args:
            num_threads (int, optional):
                The number of threads to generate datasets.
            nodes_num (int, optional):
                The number of nodes.
            data_type (str, optional):
                The data type.
            solver_type (str, optional):
                The solver type.
            train_samples_num (int, optional):
                The number of training samples.
            val_samples_num (int, optional):
                The number of validation samples.
            test_samples_num (int, optional):
                The number of test samples.
            save_path (pathlib.Path, optional):
                The save path.
            filename (str, optional):
                The filename.
            gaussian_mean_x (float, optional):
                The mean of the x-coordinate in Gaussian data generation.
            gaussian_mean_y (float, optional):
                The mean of the y-coordinate in Gaussian data generation.
            gaussian_std (float, optional):
                The standard deviation in Gaussian data generation.
        """
        # record variable data
        self.num_threads = num_threads
        self.nodes_num = nodes_num
        self.data_type = data_type
        self.solver = solver
        self.train_samples_num = train_samples_num
        self.val_samples_num = val_samples_num
        self.test_samples_num = test_samples_num
        self.save_path = save_path
        self.filename = filename
        
        # special for demand and capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        
        # special for gaussian
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std
        
        # only instance for us
        self.only_instance_for_us = only_instance_for_us
        self.check_data_type()
        
        # generate and solve
        if only_instance_for_us == False:
            # check the input variables
            self.sample_types = ["train", "val", "test"]
            self.check_num_threads()    
            self.check_solver()
            self.get_filename()

    def check_num_threads(self):
        self.samples_num = 0
        for sample_type in self.sample_types:
            self.samples_num += getattr(self, f"{sample_type}_samples_num")
            if self.samples_num % self.num_threads != 0:
                message = "``samples_num`` must be divisible by the number of threads. "
                raise ValueError(message)

    def check_data_type(self):
        generate_func_dict = {
            "uniform": self.generate_uniform,
            "gaussian": self.generate_gaussian,
        }
        supported_data_type = generate_func_dict.keys()
        if self.data_type not in supported_data_type:
            message = (
                f"The input data_type ({self.data_type}) is not a valid type, "
                f"and the generator only supports {supported_data_type}."
            )
            raise ValueError(message)
        self.generate_func = generate_func_dict[self.data_type]

    def check_solver(self):
        # get solver
        if isinstance(self.solver, SOLVER_TYPE):
            self.solver_type = self.solver
            supported_solver_dict = {
                SOLVER_TYPE.HGS: CVRPHGSSolver,
                SOLVER_TYPE.LKH: CVRPLKHSolver,
                SOLVER_TYPE.PYVRP: CVRPPyVRPSolver
            }
            supported_solver_type = supported_solver_dict.keys()
            if self.solver_type not in supported_solver_type:
                message = (
                    f"The input solver_type ({self.solver_type}) is not a valid type, "
                    f"and the generator only supports {supported_solver_type}."
                )
                raise ValueError(message)
            self.solver = supported_solver_dict[self.solver_type]()
        else:
            self.solver: CVRPSolver
            self.solver_type = self.solver.solver_type
            
        # check solver
        check_solver_dict = {
            SOLVER_TYPE.HGS: self.check_free,
            SOLVER_TYPE.LKH: self.check_lkh,
            SOLVER_TYPE.PYVRP: self.check_free,
        }
        check_func = check_solver_dict[self.solver_type]
        check_func()
    
    def check_lkh(self):
        # check if lkh is downloaded
        if shutil.which(self.solver.lkh_path) is None:
            self.download_lkh()
        # check again
        if shutil.which(self.solver.lkh_path) is None:
            message = (
                f"The LKH solver cannot be found in the path '{self.solver.lkh_path}'. "
                "Please make sure that you have entered the correct ``lkh_path``."
                "If you have not installed the LKH solver, "
                "please use function ``self.download_lkh()`` to download it."
                "Please also confirm whether the Conda environment of the terminal "
                "is consistent with the Python environment."
            )
            raise ValueError(message)

    def check_free(self):
        return
    
    def download_lkh(self):
        # download
        import wget
        lkh_url = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz"
        wget.download(url=lkh_url, out="LKH-3.0.7.tgz")
        # tar .tgz file
        os.system("tar xvfz LKH-3.0.7.tgz")
        # build LKH
        ori_dir = os.getcwd()
        os.chdir("LKH-3.0.7")
        os.system("make")
        # move LKH to the bin dir
        target_dir = os.path.join(sys.prefix, "bin")
        os.system(f"cp LKH {target_dir}")
        os.chdir(ori_dir)
        # delete .tgz file
        os.remove("LKH-3.0.7.tgz")
        shutil.rmtree("LKH-3.0.7")
    
    def get_filename(self):
        self.filename = (
            f"cvrp{self.nodes_num}_{self.data_type}"
            if self.filename is None
            else self.filename
        )
        self.file_save_path = os.path.join(self.save_path, self.filename + ".txt")
        for sample_type in self.sample_types:
            setattr(
                self,
                f"{sample_type}_file_save_path",
                os.path.join(self.save_path, self.filename + f"_{sample_type}.txt"),
            )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def generate_only_instance_for_us(self, samples: int) -> Sequence[np.ndarray]:
        self.num_threads = samples
        batch_depots_coord, batch_nodes_coord = self.generate_func()
        batch_demands = self.generate_demands()
        batch_capacities = self.generate_capacities()
        self.solver.from_data(
            depots=batch_depots_coord,
            points=batch_nodes_coord,
            demands=batch_demands,
            capacities=batch_capacities
        )
        return (
            self.solver.depots, self.solver.points, 
            self.solver.demands, self.solver.capacities
        )

    def generate(self):
        start_time = time.time()
        for _ in tqdm(
            range(self.samples_num // self.num_threads),
            desc=f"Solving CVRP Using {self.solver_type}",
        ):
            # call generate_func to generate data
            batch_depots_coord, batch_nodes_coord = self.generate_func()
            batch_demands = self.generate_demands()
            batch_capacities = self.generate_capacities()
            
            # solve
            tours = self.solver.solve(
                depots=batch_depots_coord,
                points=batch_nodes_coord,
                demands=batch_demands,
                capacities=batch_capacities.reshape(-1),
                num_threads=self.num_threads
            )

            # write to txt
            with open(self.file_save_path, "a+") as f:
                for idx, tour in enumerate(tours):
                    depot = batch_depots_coord[idx]
                    points = batch_nodes_coord[idx]
                    demands = batch_demands[idx]
                    capicity = batch_capacities[idx][0]
                    f.write("depots " + str(" ").join(str(depot_coord) for depot_coord in depot))
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
                    f.write(str(" ").join(str(node_idx) for node_idx in tour))
                    f.write("\n")
            f.close()
        
        # info
        end_time = time.time() - start_time
        print(
            f"Completed generation of {self.samples_num} samples of CVRP{self.nodes_num}."
        )
        print(f"Total time: {end_time/60:.1f}m")
        print(f"Average time: {end_time/self.samples_num:.1f}s")
        self.devide_file()

    def devide_file(self):
        with open(self.file_save_path, "r") as f:
            data = f.readlines()
        train_end_idx = self.train_samples_num
        val_end_idx = self.train_samples_num + self.val_samples_num
        train_data = data[:train_end_idx]
        val_data = data[train_end_idx:val_end_idx]
        test_data = data[val_end_idx:]
        data = [train_data, val_data, test_data]
        for sample_type, data_content in zip(self.sample_types, data):
            filename = getattr(self, f"{sample_type}_file_save_path")
            with open(filename, "w") as file:
                file.writelines(data_content)

    def generate_demands(self) -> np.ndarray:
        return np.random.randint(
            low=self.min_demand,
            high=self.max_demand,
            size=(self.num_threads, self.nodes_num)
        )
        
    def generate_capacities(self) -> np.ndarray:
        if self.min_capacity == self.max_capacity:
            return np.ones(shape=(self.num_threads, 1)) * self.min_capacity
        return np.random.randint(
            low=self.min_capacity,
            high=self.max_capacity,
            size=(self.num_threads, 1)
        )
    
    def generate_uniform(self) -> np.ndarray:
        depots = np.random.random([self.num_threads, 2])
        points = np.random.random([self.num_threads, self.nodes_num, 2]) 
        return depots, points

    def generate_gaussian(self) -> np.ndarray:
        depots = np.random.normal(
            loc=[self.gaussian_mean_x, self.gaussian_mean_y],
            scale=self.gaussian_std,
            size=(self.num_threads, 2),
        )
        points = np.random.normal(
            loc=[self.gaussian_mean_x, self.gaussian_mean_y],
            scale=self.gaussian_std,
            size=(self.num_threads, self.nodes_num, 2),
        )
        return depots, points
