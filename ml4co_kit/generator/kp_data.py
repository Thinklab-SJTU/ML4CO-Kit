import shutil
import pathlib
import numpy as np
from typing import Union, Sequence, Iterable
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.solver import KPSolver, KPORSolver


class KPDataGenerator(GeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        items_num: int = 100,
        data_type: str = "uniform",
        solver: Union[SOLVER_TYPE, KPSolver] = SOLVER_TYPE.ORTOOLS,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/kp",
        filename: str = None,
        # args for weights, values, and capacities
        sample_scale: float = 1e6,
        min_capacity: int = 25,
        max_capacity: int = 25,
    ):
        # filename
        if filename is None:
            filename = f"kp{items_num}_{data_type}"

        generate_func_dict = {
            "uniform": self._generate_uniform,
        }
        supported_solver_dict = {
            SOLVER_TYPE.ORTOOLS: KPORSolver,
        }
        check_solver_dict = {
            SOLVER_TYPE.ORTOOLS: self._check_free,
        }

        # super args
        super(KPDataGenerator, self).__init__(
            only_instance_for_us=only_instance_for_us,
            num_threads=num_threads,
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
        self.solver: KPSolver

        # args for weights, values, and capacities
        if data_type == "uniform":
            self.scale = sample_scale
            self.min_weights = 1
            self.max_weights = sample_scale
            self.min_values = 1
            self.max_values = sample_scale
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.items_num = items_num

    ##################################
    #         Generate Funcs         #
    ##################################

    def _generate_weights(self) -> np.ndarray:
        return np.random.randint(
            low=self.min_weights,
            high=self.max_weights + 1,
            size=(self.num_threads, self.items_num)
        ) / self.scale
    
    def _generate_values(self) -> np.ndarray:
        return np.random.randint(
            low=self.min_values,
            high=self.max_values + 1,
            size=(self.num_threads, self.items_num)
        ) / self.scale
        
    def _generate_capacities(self) -> np.ndarray:
        if self.min_capacity == self.max_capacity:
            return np.ones(shape=(self.num_threads,)) * self.min_capacity
        return np.random.randint(
            low=self.min_capacity,
            high=self.max_capacity,
            size=(self.num_threads,)
        )
    
    def _generate_uniform(self) -> Iterable[np.ndarray]:
        weights = self._generate_weights()
        values = self._generate_values()
        capacities = self._generate_capacities()
        return weights, values, capacities
    
    ##################################
    #      Solver-Checking Funcs     #
    ##################################
    
    def _check_free(self):
        return
    
    ##################################
    #      Data-Generating Funcs     #
    ##################################
    
    def generate_only_instance_for_us(self, samples: int) -> Sequence[np.ndarray]:
        self.num_threads = samples
        batch_weights, batch_values, batch_capacities = self.generate_func()
        self.solver.from_data(
            weights=batch_weights,
            values=batch_values,
            capacities=batch_capacities
        )
        return self.solver.weights, self.solver.values, self.solver.capacities

    def _generate_core(self):
        # call generate_func to generate data
        batch_weights, batch_values, batch_capacities = self.generate_func()

        # solve
        items_label = self.solver.solve(
            weights=batch_weights,
            values=batch_values,
            capacities=batch_capacities.reshape(-1),
            num_threads=self.num_threads
        )

        # write to txt
        with open(self.file_save_path, "a+") as f:
            for idx, vars in enumerate(items_label):
                capacity = batch_capacities[idx]
                weights = batch_weights[idx]
                values = batch_values[idx]
                f.write("weights " + str(" ").join(str(weight) for weight in weights))
                f.write(" values " + str(" ").join(str(value) for value in values))
                f.write(" capacity " + str(capacity))
                f.write(str(" label "))
                f.write(str(" ").join(str(var) for var in vars))
                f.write("\n")