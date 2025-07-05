import pathlib
import numpy as np
from typing import Union, Sequence, Iterable
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.solver import PCTSPSolver, PCTSPORSolver, PCTSPILSSolver

MAX_LENGTHS = {
    20: 2.,
    50: 3.,
    100: 4.
}

class PCTSPDataGenerator(GeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num: int = 50,
        data_type: str = "uniform",
        solver: Union[SOLVER_TYPE, PCTSPSolver] = SOLVER_TYPE.ILS,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/pctsp",
        filename: str = None,
        # args for PCTSP
        max_length_dict: dict = None,
        penalty_factor: int = 3
    ):
        # filename
        if filename is None:
            filename = f"pctsp{nodes_num}_{data_type}"

        generate_func_dict = {
            "uniform": self._generate_uniform,
        }
        supported_solver_dict = {
            SOLVER_TYPE.ORTOOLS: PCTSPORSolver,
            SOLVER_TYPE.ILS: PCTSPILSSolver,
        }
        check_solver_dict = {
            SOLVER_TYPE.ORTOOLS: self._check_free,
            SOLVER_TYPE.ILS: self._check_free,
        }

        # super args
        super(PCTSPDataGenerator, self).__init__(
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
        self.solver: PCTSPSolver

        self.nodes_num = nodes_num
        self.max_length_dict = max_length_dict if max_length_dict is not None else MAX_LENGTHS
        self.penalty_factor = penalty_factor

    ##################################
    #         Generate Funcs         #
    ##################################
    
    def _generate_depots(self) -> np.ndarray:
        return np.random.uniform(size=(self.num_threads, 2))
        
    def _generate_locs(self) -> np.ndarray:
        return np.random.uniform(size=(self.num_threads, self.nodes_num, 2))
        
    def _generate_penalties(self) -> np.ndarray:
        if self.nodes_num not in self.max_length_dict:
            raise ValueError(f"Unsupported nodes number: {self.nodes_num}. Supported: {list(self.max_length_dict.keys())}")
        penalty_max = self.max_length_dict[self.nodes_num] * (self.penalty_factor) / float(self.nodes_num)
        return np.random.uniform(size=(self.num_threads, self.nodes_num)) * penalty_max
    
    def _generate_deterministic_prizes(self) -> np.ndarray:
        # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
        # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
        # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
        return np.random.uniform(size=(self.num_threads, self.nodes_num)) * 4 / float(self.nodes_num)
    
    def _generate_stochastic_prizes(self, deterministic_prizes: np.ndarray) -> np.ndarray:
        # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
        # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
        # stochastic prize is only revealed once the node is visited
        # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
        return np.random.uniform(size=(self.num_threads, self.nodes_num)) * deterministic_prizes * 2
    
    def _generate_uniform(self) -> Iterable[np.ndarray]:
        depots = self._generate_depots()
        locs = self._generate_locs()
        penalties = self._generate_penalties()
        deterministic_prizes = self._generate_deterministic_prizes()
        stochastic_prizes = self._generate_stochastic_prizes(deterministic_prizes)
        return depots, locs, penalties, deterministic_prizes, stochastic_prizes
    
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
        depots, locs, penalties, deterministic_prizes, stochastic_prizes = self.generate_func()
        self.solver.from_data(
            depots=depots,
            points=locs,
            penalties=penalties,
            deterministic_prizes=deterministic_prizes,
            stochastic_prizes=stochastic_prizes,
        )
        return self.solver.depots, self.solver.points, self.solver.penalties, self.solver.deterministic_prizes, self.solver.stochastic_prizes

    def _generate_core(self):
        # call generate_func to generate data
        depots, locs, penalties, deterministic_prizes, stochastic_prizes = self.generate_func()

        # solve
        items_label = self.solver.solve(
            depots=depots,
            points=locs,
            penalties=penalties,
            prizes=deterministic_prizes,
            num_threads=self.num_threads
        )

        # write to txt
        with open(self.file_save_path, "a+") as f:
            for idx, tour in enumerate(items_label[1]):
                depot = depots[idx]
                loc = locs[idx]
                penalty = penalties[idx]
                deterministic_prize = deterministic_prizes[idx]
                stochastic_prize = stochastic_prizes[idx]
                f.write(f"depot {depot[0]} {depot[1]} ")
                f.write("points ")
                for i in range(len(loc)):
                    f.write(f"{loc[i][0]} {loc[i][1]} ")
                f.write("penalties ")
                for i in range(len(penalty)):
                    f.write(f"{penalty[i]} ")
                f.write(f"deterministic_prizes ")
                for i in range(len(deterministic_prize)):
                    f.write(f"{deterministic_prize[i]} ")
                f.write(f"stochastic_prizes ")
                for i in range(len(stochastic_prize)):
                    f.write(f"{stochastic_prize[i]} ")
                f.write("tours ")
                for node_idx in tour:
                    f.write(f"{node_idx} ")
                f.write("\n")
                