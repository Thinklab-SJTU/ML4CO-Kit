import os
import numpy as np
import pandas as pd
from data4co.data.tsp.tsplib_origin import TSPLIBOriginDataset
from data4co.solver.tsp.base import TSPSolver


class TSPLIBOriginEvaluator:
    def __init__(self) -> None:
        self.dataset = TSPLIBOriginDataset()
        self.support = self.dataset.support["resolved"]
        
    def evaluate(
        self,
        solver: TSPSolver,
        norm: str="EUC_2D",
        normalize: bool=False,
        **solver_args
    ):
        # record
        solved_costs = dict()
        gt_costs = dict()
        gaps = dict()
        
        # get the evaluate files' dir and the problem name list
        evaluate_dir = self.support[norm]["path"]
        solution_dir = self.support[norm]["solution"]
        problem_list = self.support[norm]["problem"]
        
        # solve
        for problem in problem_list:
            # read
            file_path = os.path.join(evaluate_dir, problem+".tsp")
            gt_tour_path = os.path.join(solution_dir, problem+".opt.tour") 
            solver.from_tsp(file_path, norm, normalize)
            solver.read_gt_tours_from_opt_tour(gt_tour_path)
            solver.solve(norm=norm, normalize=normalize, **solver_args)
            solved_cost, gt_cost, gap, _ = solver.evaluate(caculate_gap=True)
            # record
            solved_costs[problem] = solved_cost
            gt_costs[problem] = gt_cost
            gaps[problem] = gap
            
        # average
        np_solved_costs = np.array(list(solved_costs.values()))
        np_gt_costs = np.array(list(gt_costs.values()))
        np_gaps = np.array(list(gaps.values()))
        avg_solved_cost = np.average(np_solved_costs)
        avg_gt_cost = np.average(np_gt_costs)
        avg_gap = np.average(np_gaps)
        solved_costs["AVG"] = avg_solved_cost
        gt_costs["AVG"] = avg_gt_cost
        gaps["AVG"] = avg_gap
        
        # output
        return_dict = {
            "solved_costs": solved_costs,
            "gt_costs": gt_costs,
            "gaps": gaps 
        }
        df = pd.DataFrame(return_dict)
        return df
            