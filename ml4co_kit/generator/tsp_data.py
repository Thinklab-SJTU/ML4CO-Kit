import os
import shutil
import pathlib
import itertools
import numpy as np
from typing import Union
from multiprocessing import Pool
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.evaluate.tsp.base import TSPEvaluator
from ml4co_kit.generator.base import EdgeGeneratorBase
from ml4co_kit.solver import (
    TSPSolver, TSPLKHSolver, TSPConcordeSolver, TSPConcordeLargeSolver,
    TSPGAEAXSolver, TSPGAEAXLargeSolver
)
import warnings

warnings.filterwarnings("ignore")


class TSPDataGenerator(EdgeGeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num: int = 50,
        data_type: str = "uniform",
        solver: Union[SOLVER_TYPE, TSPSolver, str] = SOLVER_TYPE.LKH,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/tsp",
        filename: str = None,
        # special args for gaussian
        gaussian_mean_x: float = 0.0,
        gaussian_mean_y: float = 0.0,
        gaussian_std: float = 1.0,
        # special args for cluster
        cluster_nums: int = 10,
        cluster_std: float = 0.1,
        # special args for regret
        regret: bool = False,
        regret_save_path: str = None,
        regret_solver: TSPSolver = None,
    ):
        # filename
        if filename is None:
            filename = f"tsp{nodes_num}_{data_type}"

        # special args for gaussian
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std
        
        # special args for cluster
        self.cluster_nums = cluster_nums
        self.cluster_std = cluster_std
        
        # special args for regret
        self.regret = regret
        self.regret_cnt = 0
        self.regret_save_path = regret_save_path
        self.regret_solver = regret_solver

        # re-define
        generate_func_dict = {
            "uniform": self._generate_uniform,
            "gaussian": self._generate_gaussian,
            "cluster": self._generate_cluster,
        }
        supported_solver_dict = {
            SOLVER_TYPE.CONCORDE: TSPConcordeSolver,
            SOLVER_TYPE.LKH: TSPLKHSolver, 
            SOLVER_TYPE.CONCORDE_LARGE: TSPConcordeLargeSolver,
            SOLVER_TYPE.GA_EAX: TSPGAEAXSolver, 
            SOLVER_TYPE.GA_EAX_LARGE: TSPGAEAXLargeSolver 
        }
        check_solver_dict = {
            SOLVER_TYPE.CONCORDE: self._check_concorde,
            SOLVER_TYPE.CONCORDE_LARGE: self._check_concorde,
            SOLVER_TYPE.GA_EAX: self._check_free,
            SOLVER_TYPE.GA_EAX_LARGE: self._check_free,
            SOLVER_TYPE.LKH: self._check_lkh
        }
        
        # super args
        super(TSPDataGenerator, self).__init__(
            only_instance_for_us=only_instance_for_us,
            num_threads=num_threads,
            nodes_num=nodes_num,
            data_type=data_type,
            solver=solver,
            train_samples_num=train_samples_num,
            val_samples_num=val_samples_num,
            test_samples_num=test_samples_num,
            save_path=save_path,
            filename=filename,
            generate_func_dict=generate_func_dict,
            supported_solver_dict=supported_solver_dict,
            check_solver_dict=check_solver_dict
        )
        self.solver: TSPSolver
        self._check_regret()
        
    ##################################
    #         Generate Funcs         #
    ##################################

    def _generate_uniform(self) -> np.ndarray:
        return np.random.random([self.num_threads, self.nodes_num, 2])

    def _generate_gaussian(self) -> np.ndarray:
        return np.random.normal(
            loc=[self.gaussian_mean_x, self.gaussian_mean_y],
            scale=self.gaussian_std,
            size=(self.num_threads, self.nodes_num, 2),
        )

    def _generate_cluster(self):
        nodes_coords = np.zeros([self.num_threads, self.nodes_num, 2])
        for i in range(self.num_threads):
            cluster_centers = np.random.random([self.cluster_nums, 2])
            cluster_points = []
        for center in cluster_centers:
            points = np.random.normal(
                loc=center,
                scale=self.cluster_std,
                size=(self.nodes_num // self.cluster_nums, 2),
            )
            cluster_points.append(points)
        nodes_coords[i] = np.concatenate(cluster_points, axis=0)
        return nodes_coords
    
    ##################################
    #      Solver-Checking Funcs     #
    ################################## 
            
    def _check_lkh(self):
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

    def _check_concorde(self):
        try:
            from ml4co_kit.solver.tsp.pyconcorde import TSPConSolver
        except:
            self._recompile_concorde()

    def _recompile_concorde(self):
        concorde_path = pathlib.Path(__file__).parent.parent / "solver/tsp/pyconcorde"
        ori_dir = os.getcwd()
        os.chdir(concorde_path)
        os.system("python ./setup.py build_ext --inplace")
        os.chdir(ori_dir)
        
    def _check_free(self):
        return

    def _check_regret(self):
        if not self.regret:
            return
        if self.regret_save_path is None:
            self.regret_save_path = os.path.join(self.save_path, "regret")
        if self.regret_solver is None:
            self.regret_solver = TSPLKHSolver(lkh_max_trials=10)
        if not os.path.exists(self.regret_save_path):
            os.makedirs(self.regret_save_path)

    ##################################
    #      Data-Generating Funcs     #
    ##################################
    
    def generate_only_instance_for_us(self, samples: int) -> np.ndarray:
        self.num_threads = samples
        points = self.generate_func()
        self.solver.from_data(points=points)
        return self.solver.points

    def _generate_core(self):
        # call generate_func to generate data
        batch_nodes_coord = self.generate_func()
        
        # solve
        tours = self.solver.solve(
            points=batch_nodes_coord, num_threads=self.num_threads
        )

        # deal with regret
        if self.regret:
            if self.num_threads == 1:
                self._generate_regret(tours[0], batch_nodes_coord[0], self.regret_cnt)    
            else:
                with Pool(self.num_threads) as p2:
                    p2.starmap(
                        self._generate_regret,
                        [
                            (tour, batch_nodes_coord[idx], self.regret_cnt + idx)
                            for idx, tour in enumerate(tours)
                        ],
                    )
            self.regret_cnt += self.num_threads
            
        # write to txt
        for idx, tour in enumerate(tours):
            tour = tour[:-1]
            if (np.sort(tour) == np.arange(self.nodes_num)).all():
                with open(self.file_save_path, "a+") as f:
                    f.write(
                        " ".join(
                            str(x) + str(" ") + str(y)
                            for x, y in batch_nodes_coord[idx]
                        )
                    )
                    f.write(str(" ") + str("output") + str(" "))
                    f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                    f.write(str(" ") + str(tour[0] + 1) + str(" "))
                    f.write("\n")
                f.close()

    def _generate_regret(self, tour: np.ndarray, nodes_coord: np.ndarray, cnt: int):
        opt_tour = list(tour) + [0]
        reg_mat = self.calc_regret(nodes_coord, opt_tour)
        np.save(os.path.join(self.regret_save_path, f"{cnt}.npy"), reg_mat)

    def calc_regret(self, points: np.ndarray, opt_tour: list):
        num_nodes = points.shape[0]
        reg_mat = np.zeros((num_nodes, num_nodes))
        eva = TSPEvaluator(points)
        for i, j in itertools.combinations(range(num_nodes), 2):
            tour = self.regret_solver.regret_solve(points=points, fixed_edges=(i, j))
            cost = eva.evaluate(tour)
            opt_cost = eva.evaluate(opt_tour)
            regret = (cost - opt_cost) / opt_cost
            reg_mat[i, j] = reg_mat[j, i] = regret
        return reg_mat

    def devide_file(self):
        # basic
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
                
        # deal with regret
        if self.regret:
            for root, _, file in os.walk(self.regret_save_path):
                file.sort(key=lambda x: int(x.split(".")[0]))
                for i, reg_file in enumerate(file):
                    if i < self.train_samples_num:
                        shutil.move(
                            os.path.join(root, reg_file),
                            os.path.join(self.regret_save_path, f"train_{i}.npy"),
                        )
                    elif i < self.train_samples_num + self.val_samples_num:
                        shutil.move(
                            os.path.join(root, reg_file),
                            os.path.join(
                                self.regret_save_path, f"val_{i-train_end_idx}.npy"
                            ),
                        )
                    else:
                        shutil.move(
                            os.path.join(root, reg_file),
                            os.path.join(
                                self.regret_save_path, f"test_{i-val_end_idx}.npy"
                            ),
                        )
                break