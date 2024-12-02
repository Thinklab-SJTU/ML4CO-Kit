import os
import numpy as np
import pandas as pd
from ml4co_kit.utils.time_utils import Timer
from ml4co_kit.solver.tsp.base import TSPSolver
from ml4co_kit.data.tsp.tsplib4ml import TSPLIB4MLDataset


class TSPLIB4MLEvaluator:
    def __init__(self) -> None:
        self.dataset = TSPLIB4MLDataset()
        self.support = self.dataset.support["resolved"]

    def evaluate(
        self,
        solver: TSPSolver,
        normalize: bool = True,
        min_nodes_num: int = 51,
        max_nodes_num: int = 1002,
        show_time: bool = False,
        **solver_args
    ):
        # timer
        timer = Timer(apply=show_time)
        timer.start()
        
        # record
        solved_costs = dict()
        ref_costs = dict()
        gaps = dict()

        # get the evaluate files' dir and the problem name list
        if normalize:
            evaluate_dir = self.support["txt_normalize"]
        else:
            evaluate_dir = self.support["txt_raw"]
        problem_list = self.support["problem"]
               
        # solve
        for problem, nodes_num in problem_list:
            # check the nodes_num
            if nodes_num > max_nodes_num or nodes_num < min_nodes_num:
                continue
            
            # read problem
            problem: str
            file_path = os.path.join(evaluate_dir, problem + ".txt")
            solver.from_txt(file_path, ref=True)
            
            # real solve
            solver.solve(**solver_args)
            solved_cost, ref_cost, gap, _ = solver.evaluate(calculate_gap=True)
            
            # record
            solved_costs[problem] = solved_cost
            ref_costs[problem] = ref_cost
            gaps[problem] = gap

        # timer
        timer.end()
        timer.show_time()
        
        # average
        np_solved_costs = np.array(list(solved_costs.values()))
        np_ref_costs = np.array(list(ref_costs.values()))
        np_gaps = np.array(list(gaps.values()))
        avg_solved_cost = np.average(np_solved_costs)
        avg_ref_cost = np.average(np_ref_costs)
        avg_gap = np.average(np_gaps)
        solved_costs["AVG"] = avg_solved_cost
        ref_costs["AVG"] = avg_ref_cost
        gaps["AVG"] = avg_gap

        # output
        return_dict = {
            "solved_costs": solved_costs,
            "ref_costs": ref_costs,
            "gaps": gaps,
        }
        df = pd.DataFrame(return_dict)
        return df
