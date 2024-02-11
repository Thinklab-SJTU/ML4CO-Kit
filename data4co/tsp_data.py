import os
import lkh
import sys
import time
import shutil
import numpy as np
import tsplib95
import pathlib
from tqdm import tqdm
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")


class TSPDataGenerator:
    def __init__(
        self,
        batch_size: int=1,
        nodes_num: int=50,
        data_type: str="uniform",
        solver_type: str="lkh",
        train_samples_num: int=128000,
        val_samples_num: int=1280,
        test_samples_num: int=1280,
        save_path: pathlib.Path="data/tsp/uniform",
        filename: str=None,
        # special for gaussian
        gaussian_mean_x: float=0.0,
        gaussian_mean_y: float=0.0,
        gaussian_std: float=1.0,
        # special for lkh
        lkh_max_trials: int=1000,
        lkh_path: pathlib.Path="LKH",
        lkh_scale: int=1e6,
        lkh_runs: int=10,
        # special for concorde
        concorde_scale: int=1e6
    ):
        """_summary_

        Args:
            batch_size (int, optional): 
                The batch size. Defaults to 1.
            nodes_num (int, optional): 
                The number of nodes. Defaults to 50.
            data_type (str, optional): 
                The data type. Defaults to "uniform".
            solver_type (str, optional): 
                The solver type. Defaults to "lkh".
            train_samples_num (int, optional): 
                The number of training samples. Defaults to 128000.
            val_samples_num (int, optional): 
                The number of validation samples. Defaults to 1280.
            test_samples_num (int, optional): 
                The number of test samples. Defaults to 1280.
            save_path (pathlib.Path, optional): 
                The save path. Defaults to "data/tsp/uniform".
            filename (str, optional): 
                The filename. Defaults to None.
            gaussian_mean_x (float, optional): The mean of the x-coordinate 
                in Gaussian data generation. Defaults to 0.0.
            gaussian_mean_y (float, optional): The mean of the y-coordinate 
                in Gaussian data generation. Defaults to 0.0.
            gaussian_std (float, optional): The standard deviation in Gaussian 
                data generation. Defaults to 1.0.
            lkh_max_trials (int, optional): The maximum number of trials for 
                the LKH solver. Defaults to 1000.
            lkh_path (pathlib.Path, optional): The path to the LKH solver. 
                Defaults to "LKH".
            lkh_scale (int, optional): The scale factor for coordinates in the 
                LKH solver. Defaults to 1e6.
            lkh_runs (int, optional): The number of runs for the LKH solver. 
                Defaults to 10.
            concorde_scale (int, optional): The scale factor for coordinates 
                in the Concorde solver. Defaults to 1e6.
        """
        # record variable data
        self.batch_size = batch_size
        self.nodes_num = nodes_num
        self.data_type = data_type
        self.solver_type = solver_type
        self.train_samples_num = train_samples_num
        self.val_samples_num = val_samples_num
        self.test_samples_num = test_samples_num
        self.save_path = save_path
        self.filename = filename
        # special for gaussian
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std
        # special for lkh
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_scale = lkh_scale
        self.lkh_runs = lkh_runs
        # special for concorde
        self.concorde_scale = concorde_scale
        # check the input variables
        self.sample_types = ['train', 'val', 'test']
        self.check_batch_size()
        self.check_data_type()
        self.check_solver_type()
        self.get_filename()
        
    def check_batch_size(self):
        self.samples_num = 0
        for sample_type in self.sample_types:
            self.samples_num += getattr(self, f"{sample_type}_samples_num")
            if self.samples_num % self.batch_size != 0:
                message = f"The {self.sample_types}_samples_num must be "
                message += "evenly divided by the batch size"
                raise ValueError(message)
    
    def check_data_type(self):
        generate_func_dict = {
            "uniform": self.generate_uniform, 
            "gaussian": self.generate_gaussian
        }
        supported_data_type = generate_func_dict.keys()
        if self.data_type not in supported_data_type:
            message = f"The input data_type({self.data_type}) is not a valid type, "
            message += f"and the generator only supports {supported_data_type}"
            raise ValueError(message)
        self.generate_func = generate_func_dict[self.data_type]
        
    def check_solver_type(self):
        supported_solver_dict = {
            "lkh": self.solve_by_lkh, 
            "concorde": self.solve_by_concorde
        }
        check_solver_dict = {
            "lkh": self.check_lkh,
            "concorde": self.check_concorde
        }
        supported_solver_type = supported_solver_dict.keys()
        if self.solver_type not in supported_solver_type:
            message = f"The input solver_type({self.solver_type}) is not a valid type, "
            message += f"and the generator only supports {supported_solver_type}"
            raise ValueError(message)
        self.solver = supported_solver_dict[self.solver_type]
        check_func = check_solver_dict[self.solver_type]
        check_func()
        
    def check_lkh(self):
        # check if lkh is downloaded 
        if shutil.which(self.lkh_path) is None:
            self.download_lkh()
        # check again
        if shutil.which(self.lkh_path) is None:
            message = f"The LKH solver cannot be found in the path '{self.lkh_path}'. "
            message += "Please verify that the input lkh_path is correct. "
            message += "If you have not installed the LKH solver, "
            message += "please use the 'self.download_lkh()' function to download it."
            message += "If you are sure that the installation is correct, "
            message += "please confirm whether the Conda environment of the terminal "
            message += "is consistent with the Python environment."
            raise ValueError(message)
    
    def check_concorde(self):
        try:
            from data4co.solver import TSPConSolver
        except:
            self.recompile_concorde()

    def recompile_concorde(self):
        ori_dir = os.getcwd()
        os.chdir('solver/tsp/pyconcorde')
        os.system("python ./setup.py build_ext --inplace")
        os.chdir(ori_dir)
    
    def download_lkh(self):
        import wget
        lkh_url = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz"
        wget.download(url=lkh_url, out="LKH-3.0.7.tgz")
        os.system("tar xvfz LKH-3.0.7.tgz")
        ori_dir = os.getcwd()
        os.chdir('LKH-3.0.7')
        os.system("make")
        target_dir = os.path.join(sys.prefix, "bin")
        os.system(f"cp LKH {target_dir}")
        os.chdir(ori_dir)
        os.remove("LKH-3.0.7.tgz")
        shutil.rmtree("LKH-3.0.7")
    
    def get_filename(self):
        self.filename = f"tsp{self.nodes_num}_{self.data_type}" \
            if self.filename is None else self.filename
        self.file_save_path = os.path.join(self.save_path, self.filename + ".txt")
        for sample_type in self.sample_types:
            setattr(
                self, f"{sample_type}_file_save_path", 
                os.path.join(self.save_path, self.filename + f"_{sample_type}.txt")
            )

    def generate(self):
        with open(self.file_save_path, "w") as f:
            start_time = time.time()
            for _ in tqdm(range(self.samples_num // self.batch_size), \
                          desc=f"Solving TSP Using {self.solver_type}"):
                batch_nodes_coord = self.generate_func()
                with Pool(self.batch_size) as p:
                    tours = p.map(
                        self.solver, 
                        [batch_nodes_coord[idx] for idx in range(self.batch_size)], 
                        self.lkh_max_trials
                    )
                for idx, tour in enumerate(tours):
                    if (np.sort(tour) == np.arange(self.nodes_num)).all():
                        f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
                        f.write(str(" ") + str('output') + str(" "))
                        f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                        f.write(str(" ") + str(tour[0] + 1) + str(" "))
                        f.write("\n")
            end_time = time.time() - start_time
            f.close()
            print(f"Completed generation of {self.samples_num} samples of TSP{self.nodes_num}.")
            print(f"Total time: {end_time/60:.1f}m")
            print(f"Average time: {end_time/self.samples_num:.1f}s")
        self.devide_file()
        
    def devide_file(self):
        with open(self.file_save_path, "r") as f:
            data = f.readlines()
        train_data = data[ : self.train_samples_num]
        val_data = data[self.train_samples_num : self.train_samples_num+self.val_samples_num]
        test_data = data[self.train_samples_num+self.val_samples_num :]
        data = [train_data, val_data, test_data]
        for sample_type, data_content in zip(self.sample_types, data):
            filename = getattr(self, f"{sample_type}_file_save_path")
            with open(filename, 'w') as file:
                file.writelines(data_content)
    
    def generate_uniform(self) -> np.ndarray:
        return np.random.random([self.batch_size, self.nodes_num, 2])

    def generate_gaussian(self) -> np.ndarray:
        return np.random.normal(
            loc=[self.gaussian_mean_x, self.gaussian_mean_y], 
            scale=self.gaussian_std, 
            size=(self.batch_size, self.nodes_num, 2)
        )
    
    def solve_by_lkh(self, points: np.ndarray) -> np.ndarray:
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'
        problem.type = 'TSP'
        problem.dimension = self.nodes_num
        problem.edge_weight_type = 'EUC_2D'
        problem.node_coords = {n + 1: points[n] * self.lkh_scale for n in range(self.nodes_num)}
        solution = lkh.solve(
            solver=self.lkh_path, 
            problem=problem, 
            max_trials=self.lkh_max_trials, 
            runs=self.lkh_runs
        )
        tour = [n - 1 for n in solution[0]]   
        np_tour = np.array(tour)
        return np_tour
    
    def solve_by_concorde(self, points: np.ndarray) -> np.ndarray:
        from data4co.solver import TSPConSolver
        solver = TSPConSolver.from_data(
            points[:, 0] * self.concorde_scale, 
            points[:, 1] * self.concorde_scale, 
            norm="GEO"
        )
        solution = solver.solve(verbose=False)
        tour = solution.tour
        np_tour = np.array(tour)
        return np_tour