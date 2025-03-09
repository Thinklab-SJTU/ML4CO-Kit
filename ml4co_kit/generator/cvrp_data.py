import shutil
import pathlib
import numpy as np
from typing import Union, Sequence
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import EdgeGeneratorBase
from ml4co_kit.solver import (
    CVRPSolver, CVRPPyVRPSolver, CVRPLKHSolver, CVRPHGSSolver
)


class CVRPDataGenerator(EdgeGeneratorBase):
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
        save_path: pathlib.Path = "data/cvrp",
        filename: str = None,
        # args for demand and capacity
        min_demand: int = 1,
        max_demand: int = 10,
        min_capacity: int = 40,
        max_capacity: int = 40,
        # special args for gaussian
        gaussian_mean_x: float = 0.0,
        gaussian_mean_y: float = 0.0,
        gaussian_std: float = 1.0,
    ):
        # filename
        if filename is None:
            filename = f"cvrp{nodes_num}_{data_type}"

        # args for demand and capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        
        # special args for gaussian
        self.gaussian_mean_x = gaussian_mean_x
        self.gaussian_mean_y = gaussian_mean_y
        self.gaussian_std = gaussian_std
        
        # re-define
        generate_func_dict = {
            "uniform": self._generate_uniform,
            "gaussian": self._generate_gaussian,
        }
        supported_solver_dict = {
            SOLVER_TYPE.HGS: CVRPHGSSolver,
            SOLVER_TYPE.LKH: CVRPLKHSolver,
            SOLVER_TYPE.PYVRP: CVRPPyVRPSolver
        }
        check_solver_dict = {
            SOLVER_TYPE.HGS: self._check_free,
            SOLVER_TYPE.LKH: self._check_lkh,
            SOLVER_TYPE.PYVRP: self._check_free,
        }

        # super args
        super(CVRPDataGenerator, self).__init__(
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
        self.solver: CVRPSolver

    ##################################
    #         Generate Funcs         #
    ##################################

    def _generate_demands(self) -> np.ndarray:
        return np.random.randint(
            low=self.min_demand,
            high=self.max_demand,
            size=(self.num_threads, self.nodes_num)
        )
        
    def _generate_capacities(self) -> np.ndarray:
        if self.min_capacity == self.max_capacity:
            return np.ones(shape=(self.num_threads,)) * self.min_capacity
        return np.random.randint(
            low=self.min_capacity,
            high=self.max_capacity,
            size=(self.num_threads,)
        )
    
    def _generate_uniform(self) -> np.ndarray:
        depots = np.random.random([self.num_threads, 2])
        points = np.random.random([self.num_threads, self.nodes_num, 2]) 
        return depots, points

    def _generate_gaussian(self) -> np.ndarray:
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

    def _check_free(self):
        return
    
    ##################################
    #      Data-Generating Funcs     #
    ##################################
    
    def generate_only_instance_for_us(self, samples: int) -> Sequence[np.ndarray]:
        self.num_threads = samples
        batch_depots_coord, batch_nodes_coord = self.generate_func()
        batch_demands = self._generate_demands()
        batch_capacities = self._generate_capacities()
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

    def _generate_core(self):
        # call generate_func to generate data
        batch_depots_coord, batch_nodes_coord = self.generate_func()
        batch_demands = self._generate_demands()
        batch_capacities = self._generate_capacities()
        
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
                capicity = batch_capacities[idx]
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