import os
import uuid
import numpy as np
import gurobipy as gp
from typing import Union
from multiprocessing import Pool
from ml4co_kit.solver.lp.base import LPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class LPGurobiSolver(LPSolver):
    def __init__(self, time_limit: float = 60.0):
        super(LPGurobiSolver, self).__init__(
            solver_type=SOLVER_TYPE.GUROBI, time_limit=time_limit
        )
        
    def _solve(self, w: np.ndarray, c: np.ndarray, b: np.ndarray) -> np.ndarray:
        # create gurobi model
        tmp_name = uuid.uuid4().hex[:9]
        model = gp.Model(f"{tmp_name}")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("Threads", 1)
        
        # variables
        vars_num = len(c)
        model.addVars(vars_num, 1, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
        model.update()
        x = gp.MVar.fromlist(model.getVars())

        # constr.
        model.addConstr(w @ x == b)

        # object
        model.setObjective(c.T @ x, gp.GRB.MINIMIZE)

        # Solve
        model.write(f"{tmp_name}.lp")
        model.optimize()
        os.remove(f"{tmp_name}.lp")
        
        return np.array(x.x)
        
    def solve(
        self,
        w: Union[list, np.ndarray] = None,
        c: Union[list, np.ndarray] = None,
        b: Union[list, np.ndarray] = None,
        num_threads: int = 1,
        show_time: bool = False
    ) -> np.ndarray:
        # preparation
        self.from_data(w=w, c=c, b=b)
        timer = Timer(apply=show_time)
        timer.start()

        # solve
        sols = list()
        w_shape = self.w.shape
        c_shape = self.c.shape
        b_shape = self.b.shape
        num_instance = w_shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_instance, self.solve_msg, show_time):
                sols.append(self._solve(self.w[idx], self.c[idx], self.b[idx]))
        else:
            batch_w = self.w.reshape(-1, num_threads, w_shape[-2], w_shape[-1])
            batch_b = self.b.reshape(-1, num_threads, b_shape[-1])
            batch_c = self.c.reshape(-1, num_threads, c_shape[-1])
            for idx in iterative_execution(
                range, num_instance // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_sols = p1.starmap(
                        self._solve,
                        [  (batch_w[idx][inner_idx], 
                            batch_c[idx][inner_idx], 
                            batch_b[idx][inner_idx]) 
                            for inner_idx in range(num_threads)
                        ],
                    )
                for sol in cur_sols:
                    sols.append(sol)

        # format
        self.from_data(x=sols, ref=False)
        
        # show time
        timer.end()
        timer.show_time()

        return self.x  
    
    def __str__(self) -> str:
        return "LPGurobiSolver"