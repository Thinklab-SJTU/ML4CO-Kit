import os
import sys
import time
import shutil
import itertools
import numpy as np
import pathlib
from tqdm import tqdm
from typing import Union
from multiprocessing import Pool
from data4co.solver.tsp import TSPLKHSolver
from data4co.utils.tsp_utils import TSPEvaluator
from data4co.solver.tsp import TSPSolver, TSPLKHSolver, TSPConcordeSolver
import warnings
warnings.filterwarnings("ignore")


class TSPDataGenerator:
    def __init__(
        self,
        num_threads: int=1,
        nodes_num: int=50,
        data_type: str="uniform",
        solver: Union[str, TSPSolver]="lkh",
        train_samples_num: int=128000,
        val_samples_num: int=1280,
        test_samples_num: int=1280,
        save_path: pathlib.Path="data/tsp/uniform",
        filename: str=None,
        # special for gaussian
        gaussian_mean_x: float=0.0,
        gaussian_mean_y: float=0.0,
        gaussian_std: float=1.0,
        # special for cluster
        cluster_nums: int=10,
        cluster_std: float=0.1,
        # special for regret
        regret: bool=False,
        regret_save_path: str=None,
        regret_solver: TSPSolver=None
    ):
        """
        TSPDataGenerator
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
        # special for gaussian
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std
        # special for cluster
        self.cluster_nums = cluster_nums
        self.cluster_std = cluster_std
        # special for regret
        self.regret = regret
        self.regret_save_path = regret_save_path
        self.regret_solver = regret_solver
        # check the input variables
        self.sample_types = ['train', 'val', 'test']
        self.check_num_threads()
        self.check_data_type()
        self.check_solver()
        self.get_filename()
        self.check_regret()
        
    def check_num_threads(self):
        self.samples_num = 0
        for sample_type in self.sample_types:
            self.samples_num += getattr(self, f"{sample_type}_samples_num")
            if self.samples_num % self.num_threads != 0:
                message = f"The {self.sample_types}_samples_num must be "
                message += "evenly divided by the number of threads"
                raise ValueError(message)
    
    def check_data_type(self):
        generate_func_dict = {
            "uniform": self.generate_uniform, 
            "gaussian": self.generate_gaussian,
            "cluster": self.generate_cluster
        }
        supported_data_type = generate_func_dict.keys()
        if self.data_type not in supported_data_type:
            message = f"The input data_type({self.data_type}) is not a valid type, "
            message += f"and the generator only supports {supported_data_type}"
            raise ValueError(message)
        self.generate_func = generate_func_dict[self.data_type]
        
    def check_solver(self):
        # get solver
        if type(self.solver) == str:
            self.solver_type = self.solver
            supported_solver_dict = {
                "lkh": TSPLKHSolver, 
                "concorde": TSPConcordeSolver
            }
            supported_solver_type = supported_solver_dict.keys()
            if self.solver_type not in supported_solver_type:
                message = f"The input solver_type({self.solver_type}) is not a valid type, "
                message += f"and the generator only supports {supported_solver_type}"
                raise ValueError(message)
            self.solver = supported_solver_dict[self.solver_type]()
        else:
            self.solver: TSPSolver
            self.solver_type = self.solver.solver_type   
        # check solver
        check_solver_dict = {
            "lkh": self.check_lkh,
            "concorde": self.check_concorde
        }
        check_func = check_solver_dict[self.solver_type]
        check_func()
        
    def check_lkh(self):
        # check if lkh is downloaded
        if shutil.which(self.solver.lkh_path) is None:
            self.download_lkh()
        # check again
        if shutil.which(self.solver.lkh_path) is None:
            message = f"The LKH solver cannot be found in the path '{self.solver.lkh_path}'. "
            message += "Please verify that the input lkh_path is correct. "
            message += "If you have not installed the LKH solver, "
            message += "please use the 'self.download_lkh()' function to download it."
            message += "If you are sure that the installation is correct, "
            message += "please confirm whether the Conda environment of the terminal "
            message += "is consistent with the Python environment."
            raise ValueError(message)
    
    def check_concorde(self):
        try:
            from data4co.solver.tsp.pyconcorde import TSPConSolver
        except:
            self.recompile_concorde()

    def recompile_concorde(self):
        concorde_path = pathlib.Path(__file__).parent.parent / "solver/tsp/pyconcorde"
        ori_dir = os.getcwd()
        os.chdir(concorde_path)
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
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def check_regret(self):
        if not self.regret:
            return
        if self.regret_save_path is None:
            self.regret_save_path = os.path.join(self.save_path, "regret")
        if self.regret_solver is None:
            self.regret_solver = TSPLKHSolver(lkh_max_trials=10)
        if not os.path.exists(self.regret_save_path):
            os.makedirs(self.regret_save_path)
   
    def generate(self):
        with open(self.file_save_path, "w") as f:
            start_time = time.time()
            cnt = 0
            for _ in tqdm(range(self.samples_num // self.num_threads), \
                          desc=f"Solving TSP Using {self.solver_type}"):
                batch_nodes_coord = self.generate_func()
                with Pool(self.num_threads) as p1:
                    tours = p1.map(
                        self.solver.solve,
                        [batch_nodes_coord[idx] for idx in range(self.num_threads)], 
                    )
                # deal with regret
                if self.regret:
                    p1.close()  # Close the pool to indicate that no more tasks will be submitted
                    p1.join()  # Wait for all processes in the pool to complete
                    with Pool(self.num_threads) as p2:
                        p2.starmap(
                            self.generate_regret,
                            [(tour, batch_nodes_coord[idx], cnt+idx) for idx, tour in enumerate(tours)]
                        )
                    cnt += self.num_threads
                # write to txt    
                for idx, tour in enumerate(tours):
                    tour = tour[:-1]
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
    
    def generate_regret(
        self,
        tour: np.ndarray,
        nodes_coord: np.ndarray,
        cnt: int
    ):
        opt_tour = list(tour) + [0]
        reg_mat = self.calc_regret(nodes_coord, opt_tour)
        np.save(os.path.join(self.regret_save_path, f'{cnt}.npy'), reg_mat)
    
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
            with open(filename, 'w') as file:
                file.writelines(data_content)
        # deal with regret
        if self.regret:
            for root, _, file in os.walk(self.regret_save_path):
                file.sort(key=lambda x: int(x.split('.')[0]))
                for i, reg_file in enumerate(file):
                    if i < self.train_samples_num:
                        shutil.move(
                            os.path.join(root, reg_file), 
                            os.path.join(self.regret_save_path, f'train_{i}.npy'))
                    elif i < self.train_samples_num+self.val_samples_num:
                        shutil.move(
                            os.path.join(root, reg_file), 
                            os.path.join(self.regret_save_path, f'val_{i-train_end_idx}.npy')
                        )
                    else:
                        shutil.move(
                            os.path.join(root, reg_file), 
                            os.path.join(self.regret_save_path, f'test_{i-val_end_idx}.npy')
                        )
                break
    
    def generate_uniform(self) -> np.ndarray:
        return np.random.random([self.num_threads, self.nodes_num, 2])

    def generate_gaussian(self) -> np.ndarray:
        return np.random.normal(
            loc=[self.gaussian_mean_x, self.gaussian_mean_y], 
            scale=self.gaussian_std, 
            size=(self.num_threads, self.nodes_num, 2)
        )
        
    def generate_cluster(self):
        nodes_coords = np.zeros([self.num_threads, self.nodes_num, 2])
        for i in range(self.num_threads):
            cluster_centers = np.random.random([self.cluster_nums, 2]) 
            cluster_points = []
        for center in cluster_centers:
            points = np.random.normal(
                loc=center, 
                scale=self.cluster_std, 
                size=(self.nodes_num // self.cluster_nums, 2)
            )
            cluster_points.append(points)
        nodes_coords[i] = np.concatenate(cluster_points, axis=0)
        return nodes_coords

    def calc_regret(self, points: np.ndarray, opt_tour: list):
        num_nodes = points.shape[0]
        reg_mat = np.zeros((num_nodes, num_nodes))
        eva = TSPEvaluator(points)
        for i, j in itertools.combinations(range(num_nodes), 2):
            tour = self.regret_solver.regret_solve(
                points=points, 
                fixed_edges=(i, j)
            )
            cost = eva.evaluate(tour)
            opt_cost = eva.evaluate(opt_tour)
            regret = (cost - opt_cost) / opt_cost
            reg_mat[i, j] = reg_mat[j, i] = regret
        return reg_mat
