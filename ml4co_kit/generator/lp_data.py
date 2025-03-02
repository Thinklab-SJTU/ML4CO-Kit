import random
import pathlib
import numpy as np
from typing import Union, Tuple
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.solver import LPSolver, LPGurobiSolver


class LPDataGenerator(GeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        vars_num: int = 20,
        constr_num: int = 16,
        sparse_ratio: float = 0.0,
        data_type: str = "uniform",
        solver: Union[SOLVER_TYPE, LPSolver, str] = SOLVER_TYPE.GUROBI,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "dataset/lp",
        filename: str = None
    ):
        if filename is None:
            if sparse_ratio == 0:
                filename = f"lp_{data_type}_dense_{vars_num}_{constr_num}"
            else:    
                filename = f"lp_{data_type}_sparse_{sparse_ratio:.2f}_{vars_num}_{constr_num}"

        # re-define
        generate_func_dict = {
            "uniform": self._generate_uniform
        }
        supported_solver_dict = {
            SOLVER_TYPE.GUROBI: LPGurobiSolver
        }
        check_solver_dict = {
            SOLVER_TYPE.GUROBI: self._check_free
        }

        # super args
        super(LPDataGenerator, self).__init__(
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
        self.solver: LPSolver

        # special
        self.vars_num = vars_num
        self.constr_num = constr_num
        self.sparse_ratio = sparse_ratio
        self.nnz = int(self.constr_num * self.vars_num * (1-self.sparse_ratio))
        
    ##################################
    #         Generate Funcs         #
    ##################################
    
    def _generate_uniform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w_list = list()
        c_list = list()
        b_list = list()
        for _ in range(self.num_threads):
            max_trials = 10
            cur_trials = 0
            fail_flag = False
            while True:
                cur_trials += 1
                w = np.zeros((self.constr_num, self.vars_num), dtype=np.float32)
                edge_index = np.zeros((self.nnz, 2))
                edge_index_1d = random.sample(range(self.constr_num * self.vars_num), self.nnz)
                edge_feature = np.random.normal(0, 1, self.nnz)
                for l in range(self.nnz):
                    i = int(edge_index_1d[l] / self.vars_num)
                    j = edge_index_1d[l] - i * self.vars_num
                    edge_index[l, 0] = i
                    edge_index[l, 1] = j
                    w[i, j] = edge_feature[l] 
                
                # break
                if np.linalg.matrix_rank(w) == self.constr_num:
                    break
                if cur_trials == max_trials:
                    fail_flag = True
                    break
            
            # check fail flag
            if fail_flag:
                raise ValueError(
                    f"After {max_trials} attempts, the matrix with full rank is still not generated. "
                    "Please check the parameter setting and reduce the value of ``spare_ratio`` appropriately."
                )
            
            # randomly generate optimal solution for primal and dual
            x_hat = np.random.uniform(0, 1, self.vars_num)
            y_hat = np.random.uniform(0, 1, self.constr_num)
            beta_indices = np.random.choice(range(self.constr_num + self.vars_num), size=self.constr_num, replace=False)
            beta = np.zeros(self.constr_num + self.vars_num)
            beta[beta_indices] = 1
            x_hat[beta[:self.vars_num] == 0] = 0
            y_hat[beta[self.vars_num:] == 1] = 0
            b = w @ x_hat
            c = w.T @ y_hat
            w_list.append(w)
            c_list.append(c)
            b_list.append(b)
            
        return np.array(w_list), np.array(c_list), np.array(b_list)

    ##################################
    #      Data-Generating Funcs     #
    ##################################
    
    def generate_only_instance_for_us(self, samples: int) -> np.ndarray:
        self.num_threads = samples
        w, c, b = self.generate_func()
        self.solver.from_data(w=w, c=c, b=b)
        return self.solver.w, self.solver.c, self.solver.b

    def _generate_core(self):
        # call generate_func to generate data
        w, c, b = self.generate_func()
        
        # solve
        sols = self.solver.solve(w=w, c=c, b=b)
        
        # write to txt
        for _w, _c, _b, _sol in zip(w, c, b, sols):
            with open(self.file_save_path, "a+") as f:
                f.write(str("w") + str(" "))
                for line in _w:
                    f.write(" ".join(str(x) + str(" ") for x in line))
                    f.write(str(" "))
                f.write("c " + str(" ").join(str(cc) for cc in _c))
                f.write(" b " + str(" ").join(str(bb) for bb in _b))
                f.write(str(" output") + str(" "))
                f.write(str(" ").join(str(xx) for xx in _sol))
                f.write("\n")
            f.close()