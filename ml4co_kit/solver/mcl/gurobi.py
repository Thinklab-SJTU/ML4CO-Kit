import os
import uuid
import copy
import numpy as np
import gurobipy as gp
from typing import List
from multiprocessing import Pool
from ml4co_kit.solver.mcl.base import MClSolver
from ml4co_kit.utils.graph.mcl import MClGraphData
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class MClGurobiSolver(MClSolver):
    def __init__(self, weighted: bool = False, time_limit: float = 60.0):
        super(MClGurobiSolver, self).__init__(
            solver_type=SOLVER_TYPE.GUROBI, weighted=weighted, time_limit=time_limit
        )
        self.tmp_name = None

    def solve(
        self,
        graph_data: List[MClGraphData] = None,
        num_threads: int = 1,
        show_time: bool = False
    ) -> List[MClGraphData]:
        # preparation
        if graph_data is not None:
            self.graph_data = graph_data
        timer = Timer(apply=show_time)
        timer.start()
        self.tmp_name = uuid.uuid4().hex[:9]
        
        # solve
        solutions = list()
        graph_num = len(self.graph_data)
        if num_threads == 1:
            for idx in iterative_execution(range, graph_num, self.solve_msg, show_time):
                solutions.append(self._solve(idx=idx))
        else:
            for idx in iterative_execution(
                range, graph_num // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_sols = p1.map(
                        self._solve,
                        [
                            idx * num_threads + inner_idx
                            for inner_idx in range(num_threads)
                        ],
                    )
                for sol in cur_sols:
                    solutions.append(sol)
            
        # restore solutions
        self.from_graph_data(nodes_label=solutions, ref=False, cover=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        return self.graph_data
    
    def _solve(self, idx: int) -> np.ndarray:
        # graph
        mcl_graph: MClGraphData = self.graph_data[idx]
        cp_mcl_graph = copy.deepcopy(mcl_graph)
        
        # number of graph's nodes
        nodes_num = cp_mcl_graph.nodes_num
        
        # to complement remove self loop
        cp_mcl_graph.to_complement()
        cp_mcl_graph.remove_self_loop()
        
        # create gurobi model
        model = gp.Model(f"MCl-{self.tmp_name}-{idx}")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("Threads", 1)
        
        # edge list
        senders = cp_mcl_graph.edge_index[0]
        receivers = cp_mcl_graph.edge_index[1]
        edge_list = [(min([s, r]), max([s, r])) for s,r in zip(senders, receivers)]
        unique_edge_List = set(edge_list)
        
        # Constr.
        var_dict = model.addVars(nodes_num, vtype=gp.GRB.BINARY)
        for (s, r) in unique_edge_List:
            xs = var_dict[s]
            xr = var_dict[r]
            model.addConstr(xs + xr <= 1, name="e%d-%d" % (s, r))
            
        # Object
        object = gp.quicksum(-var_dict[int(n)]  for n in range(nodes_num))
        model.setObjective(object, gp.GRB.MINIMIZE)
        
        # Solve
        model.write(f"MCl-{self.tmp_name}-{self.tmp_name}-{idx}.lp")
        model.optimize()
        os.remove(f"MCl-{self.tmp_name}-{self.tmp_name}-{idx}.lp")

        # return
        return np.array([int(var_dict[key].X) for key in var_dict])
    
    def __str__(self) -> str:
        return "MClGurobiSolver"
