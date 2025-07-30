import pathlib
import numpy as np
from typing import Union, Sequence
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import EdgeGeneratorBase
from ml4co_kit.solver import SPCTSPSolver, SPCTSPReoptSolver


class SPCTSPDataGenerator(EdgeGeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num: int = 50,
        data_type: str = "uniform",
        solver: Union[SOLVER_TYPE, SPCTSPSolver] = SOLVER_TYPE.REOPT,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/spctsp",
        filename: str = None,
        # special args for uniform
        uniform_k: float = 3.0,
        uniform_prize_factor: float = 4.0, 
        uniform_penalty_factor: float = 3.0,
    ):
        # filename
        if filename is None:
            filename = f"spctsp{nodes_num}_{data_type}"

        # special args for uniform
        self.uniform_k = uniform_k
        self.uniform_prize_factor = uniform_prize_factor
        self.uniform_penalty_factor = uniform_penalty_factor

        # re-define
        generate_func_dict = {
            "uniform": self._generate_uniform,
        }
        supported_solver_dict = {
            SOLVER_TYPE.REOPT: SPCTSPReoptSolver
        }
        check_solver_dict = {
            SOLVER_TYPE.REOPT: self._check_free,
        }

        # super args
        super(SPCTSPDataGenerator, self).__init__(
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
        self.solver: SPCTSPSolver

    ##################################
    #         Generate Funcs         #
    ##################################
    
    def _generate_uniform(self) -> Sequence[np.ndarray]:
        # depots
        depots = np.random.uniform(size=(self.num_threads, 2))
        
        # points
        points = np.random.uniform(size=(self.num_threads, self.nodes_num, 2))
        
        # penalties
        penalty_max = self.uniform_penalty_factor * self.uniform_k / self.nodes_num
        penalties = penalty_max * np.random.uniform(size=(self.num_threads, self.nodes_num))
        
        # norm_prizes
        prize_max = self.uniform_prize_factor / self.nodes_num
        norm_prizes = prize_max * np.random.uniform(size=(self.num_threads, self.nodes_num))

        # stochastic_norm_prizes
        stochastic_scale = np.random.uniform(size=(self.num_threads, self.nodes_num)) * 2
        stochastic_norm_prizes = norm_prizes * stochastic_scale

        return depots, points, penalties, norm_prizes, stochastic_norm_prizes
    
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
        depots, points, penalties, norm_prizes, stochastic_norm_prizes = self.generate_func()
        self.solver.from_data(
            depots=depots, 
            points=points, 
            node_penalties=penalties, 
            norm_node_prizes=norm_prizes, 
            stochastic_norm_prizes=stochastic_norm_prizes
        )
        return (
            self.solver.depots,  
            self.solver.points, 
            self.solver.penalties, 
            self.solver.norm_prizes, 
            self.solver.stochastic_norm_prizes
        )
        
    def _generate_core(self):
        # call generate_func to generate data
        depots, points, penalties, norm_prizes, stochastic_norm_prizes = self.generate_func()
        depots: np.ndarray = depots.dtype(np.float32)
        points: np.ndarray = points.dtype(np.float32)
        penalties: np.ndarray = penalties.dtype(np.float32)
        norm_prizes: np.ndarray = norm_prizes.dtype(np.float32)
        stochastic_norm_prizes: np.ndarray = stochastic_norm_prizes.dtype(np.float32)
        
        # solve
        tours = self.solver.solve(
            depots=depots, 
            points=points, 
            penalties=penalties, 
            norm_prizes=norm_prizes,
            stochastic_norm_prizes=stochastic_norm_prizes, 
            num_threads=self.num_threads
        )
        
        # write to txt
        for idx, tour in enumerate(tours):
            with open(self.file_save_path, "a+") as f:
                cur_depot = depots[idx]
                cur_points = points[idx]
                cur_penalty = penalties[idx]
                cur_prize = norm_prizes[idx]
                cur_stoc_prize = stochastic_norm_prizes[idx]
                f.write(f"depots {cur_depot[0]} {cur_depot[1]} ")
                f.write("points ")
                for i in range(len(cur_points)):
                    f.write(f"{cur_points[i][0]} {cur_points[i][1]} ")
                f.write("penalties ")
                for i in range(len(cur_penalty)):
                    f.write(f"{cur_penalty[i]} ")
                f.write(f"norm_prizes ")
                for i in range(len(cur_prize)):
                    f.write(f"{cur_prize[i]} ")
                f.write(f"stochastic_norm_prizes ")
                for i in range(len(cur_stoc_prize)):
                    f.write(f"{cur_stoc_prize[i]} ")
                f.write("output ")
                for node_idx in tour:
                    f.write(f"{node_idx} ")
                f.write("\n")