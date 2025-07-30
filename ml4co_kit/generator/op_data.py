import pathlib
import numpy as np
from typing import Union, Sequence, Iterable
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import EdgeGeneratorBase
from ml4co_kit.solver import OPSolver, OPGurobiSolver


class OPDataGenerator(EdgeGeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num: int = 50,
        max_length: float = 3.0,
        data_type: str = "dist", # "constant", "unif", "dist"
        solver: Union[SOLVER_TYPE, OPSolver] = SOLVER_TYPE.GUROBI,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/op",
        filename: str = None,
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        # filename
        if filename is None:
            filename = f"op{nodes_num}_{data_type}"

        generate_func_dict = {
            "constant": self._generate_constant,
            "uniform": self._generate_uniform,
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
            nodes_num=nodes_num,
            data_type=data_type,
            solver=solver,
            train_samples_num=train_samples_num,
            val_samples_num=val_samples_num,
            test_samples_num=test_samples_num,
            save_path=save_path,
            filename=filename,
            precision=precision,
            generate_func_dict=generate_func_dict,
            supported_solver_dict=supported_solver_dict,
            check_solver_dict=check_solver_dict
        )
        self.solver: OPSolver
        self.max_length = max_length

    ##################################
    #         Generate Funcs         #
    ##################################
    
    def _generate_constant(self) -> np.ndarray:
        return np.ones((self.num_threads, self.nodes_num))
    
    def _generate_uniform(self) -> np.ndarray:
        return (1 + np.random.randint(0, 100, size=(self.num_threads, self.nodes_num))) / 100.
    
    def _generate_dist(self) -> np.ndarray:
        pass
    
    ##################################
    #      Solver-Checking Funcs     #
    ##################################
    
    def _check_free(self):
        return
    
    ##################################
    #      Data-Generating Funcs     #
    ##################################
    
    def _generate_batch_data(self) -> Sequence[np.ndarray]:
        depots = np.random.uniform(size=(self.num_threads, 2)).astype(self.precision)
        points = np.random.uniform(size=(self.num_threads, self.nodes_num, 2)).astype(self.precision)
        if self.data_type == "dist":
            prize_: np.ndarray = np.linalg.norm(depots[:, None, :] - points, axis=-1)
            prizes = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.
        else:
            prizes: np.ndarray = self.generate_func()
        prizes = prizes.astype(self.precision)
        max_lengths: np.ndarray = np.full(self.num_threads, self.max_length).astype(self.precision)
        return depots, points, prizes, max_lengths
    
    def generate_only_instance_for_us(self, samples: int) -> Sequence[np.ndarray]:
        self.num_threads = samples
        depots, locs, prizes, max_lengths = self._generate_batch_data()
        self.solver.from_data(
            depots=depots,
            points=locs,
            prizes=prizes,
            max_lengths=max_lengths
        )
        return self.solver.depots, self.solver.points, self.solver.prizes, self.solver.max_lengths

    def _generate_core(self):
        # call generate_func to generate data
        depots, locs, prizes, max_lengths = self._generate_batch_data()

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
            for idx, tour in enumerate(items_label[1]):
                depot = depots[idx]
                loc = locs[idx]
                prize = prizes[idx]
                max_length = max_lengths[idx]
                f.write(f"depots {depot[0]} {depot[1]} ")
                f.write("points ")
                for i in range(len(loc)):
                    f.write(f"{loc[i][0]} {loc[i][1]} ")
                f.write("prizes ")
                for i in range(len(prize)):
                    f.write(f"{prize[i]} ")
                f.write("max_length ")
                f.write(f"{max_length} ")
                f.write("tours ")
                for node_idx in tour:
                    f.write(f"{node_idx} ")
                f.write("\n")
                