r"""
OR-Tools for solving MCut.

OR-Tools is open source software for combinatorial optimization, which seeks to 
find the best solution to a problem out of a very large set of possible solutions.

We follow https://developers.google.cn/optimization.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import List
from multiprocessing import Pool
from ortools.sat.python import cp_model
from ml4co_kit.solver.mcut.base import MCutSolver
from ml4co_kit.utils.graph.mcut import MCutGraphData
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class MCutORSolver(MCutSolver):
    def __init__(self, weighted: bool = False, time_limit: float = 60.0):
        super(MCutORSolver, self).__init__(
            solver_type=SOLVER_TYPE.ORTOOLS, weighted=weighted, time_limit=time_limit
        )
        
    def solve(
        self,
        graph_data: List[MCutGraphData] = None,
        num_threads: int = 1,
        show_time: bool = False
    ) -> List[MCutGraphData]:
        # preparation
        if graph_data is not None:
            self.graph_data = graph_data
        timer = Timer(apply=show_time)
        timer.start()
        
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
        mcut_graph: MCutGraphData = self.graph_data[idx]
        
        # number of graph's nodes
        nodes_num = mcut_graph.nodes_num
        
        # remove self loop
        mcut_graph.remove_self_loop()
        
        # create model
        model = cp_model.CpModel()
        
        # edge list
        senders = mcut_graph.edge_index[0]
        receivers = mcut_graph.edge_index[1]
        edge_list = [(min([s, r]), max([s, r])) for s,r in zip(senders, receivers)]
        unique_edge_List = set(edge_list)
        
        # Vars.
        vertices = np.arange(nodes_num)
        x = {v: model.NewBoolVar(f'x_{v}') for v in vertices}
            
        # Object
        cut_edges = []
        for (u, v) in unique_edge_List:
            xor_var = model.NewBoolVar(f'xor_{u}_{v}')
            model.Add(x[u] != x[v]).OnlyEnforceIf(xor_var)
            model.Add(x[u] == x[v]).OnlyEnforceIf(xor_var.Not())
            cut_edges.append(xor_var)
        model.Maximize(sum(cut_edges))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = list()
            for v in vertices:
                solution.append(solver.BooleanValue(x[v]))
        else:
            raise ValueError("no feasible solution has been found")
    
        # solution
        sol = np.array(solution).astype(np.int32)
        return sol
    
    def __str__(self) -> str:
        return "MCutORSolver"
