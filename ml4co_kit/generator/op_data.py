import pathlib
import numpy as np
from typing import Union, Sequence, Iterable
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.solver import OPSolver, OPGurobiSolver

MAX_LENGTHS = {
    20: 2.,
    50: 3.,
    100: 4.
}

class OPDataGenerator(GeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num: int = 50,
        data_type: str = "dist", # "const", "unif", "dist"
        solver: Union[SOLVER_TYPE, OPSolver] = SOLVER_TYPE.GUROBI,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/op",
        filename: str = None,
        # args for OP
        max_length_dict: dict = None,
    ):
        # filename
        if filename is None:
            filename = f"op{nodes_num}_{data_type}"

        generate_func_dict = {
            "const": self._generate_const,
            "unif": self._generate_uniform,
            "dist": self._generate_dist,
        }
        supported_solver_dict = {
            SOLVER_TYPE.GUROBI: OPGurobiSolver,
        }
        check_solver_dict = {
            SOLVER_TYPE.GUROBI: self._check_free,
        }

        # super args
        super(OPDataGenerator, self).__init__(
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
        self.solver: OPSolver

        self.nodes_num = nodes_num
        self.max_length_dict = max_length_dict if max_length_dict is not None else MAX_LENGTHS

    ##################################
    #         Generate Funcs         #
    ##################################
    
    def _generate_depots(self) -> np.ndarray:
        return np.random.uniform(size=(self.num_threads, 2))
        
    def _generate_locs(self) -> np.ndarray:
        return np.random.uniform(size=(self.num_threads, self.nodes_num, 2))
        
    def _generate_prizes(
        self, 
        prize_type: str = "dist", 
        depots: np.ndarray = None, 
        locs: np.ndarray = None,
    ) -> np.ndarray:
        if prize_type == "const":
            prize = np.ones((self.num_threads, self.nodes_num))
        elif prize_type == "unif":
            prize = (1 + np.random.randint(0, 100, size=(self.num_threads, self.nodes_num))) / 100.
        elif prize_type == "dist":
            prize_ = np.linalg.norm(depots[:, None, :] - locs, axis=-1)
            prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.
        else:
            raise ValueError(f"Unsupported prize type: {prize_type}")
        return prize
    
    def _generate_max_lengths(self) -> np.ndarray:
        if self.nodes_num not in self.max_length_dict:
            raise ValueError(f"Unsupported nodes number: {self.nodes_num}. Supported: {list(self.max_length_dict.keys())}")
        return np.full(self.num_threads, self.max_length_dict[self.nodes_num])
    
    def _generate_const(self) -> Iterable[np.ndarray]:
        depots = self._generate_depots()
        locs = self._generate_locs()
        prizes = self._generate_prizes(prize_type="const")
        max_lengths = self._generate_max_lengths()
        return depots, locs, prizes, max_lengths
    
    def _generate_uniform(self) -> Iterable[np.ndarray]:
        depots = self._generate_depots()
        locs = self._generate_locs()
        prizes = self._generate_prizes(prize_type="unif")
        max_lengths = self._generate_max_lengths()
        return depots, locs, prizes, max_lengths
    
    def _generate_dist(self) -> Iterable[np.ndarray]:
        depots = self._generate_depots()
        locs = self._generate_locs()
        prizes = self._generate_prizes(prize_type="dist", depots=depots, locs=locs)
        max_lengths = self._generate_max_lengths()
        return depots, locs, prizes, max_lengths
    
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
        depots, locs, prizes, max_lengths = self.generate_func()
        self.solver.from_data(
            depots=depots,
            points=locs,
            prizes=prizes,
            max_lengths=max_lengths
        )
        return self.solver.depots, self.solver.points, self.solver.prizes, self.solver.max_lengths

    def _generate_core(self):
        # call generate_func to generate data
        depots, locs, prizes, max_lengths = self.generate_func()

        # solve
        items_label = self.solver.solve(
            depot=depots,
            loc=locs,
            prize=prizes,
            max_length=max_lengths,
            num_threads=self.num_threads
        )

        # write to txt
        with open(self.file_save_path, "a+") as f:
            for idx, _, tours in enumerate(items_label):
                depot = depots[idx]
                loc = locs[idx]
                prize = prizes[idx]
                max_length = max_lengths[idx]
                f.write(f"depot {depot[0]} {depot[1]} ")
                f.write("points ")
                for i in range(len(loc)):
                    f.write(f"{loc[i][0]} {loc[i][1]} ")
                f.write("prizes ")
                for i in range(len(prize)):
                    f.write(f"{prize[i]} ")
                f.write("max_length ")
                f.write(f"{max_length}\n")