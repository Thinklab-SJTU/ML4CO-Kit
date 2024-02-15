import numpy as np
import pandas as pd
from tqdm import tqdm
from data4co.data.tsp.tsplib import TSPLIBDataset, TSPLIBData
from data4co.solver.tsp.base import TSPSolver
from data4co.utils.tsp_utils import TSPEvaluator


class TSPLIBEvaluator:
    def __init__(self) -> None:
        self.dataset = TSPLIBDataset()
        self.tsplib_data = self.dataset.data
    
    def evaluate(
        self, 
        solver: TSPSolver,
        type: str="EUC_2D"
    ):
        solved_costs = dict()
        gt_costs = dict()
        gaps = dict()
        test_data = self.tsplib_data["resolved"][type]
        test_data: dict
        for name, data in tqdm(test_data.items()):
            data: TSPLIBData
            # read data
            points = data.node_coords
            gt_tour = data.tour
            solver.from_data(points)
            solver.read_gt_tours(gt_tour)
            # solve, caculate gap
            tour = solver.solve().tolist()
            eva = TSPEvaluator(points)
            gt_cost = eva.evaluate(gt_tour)
            solved_cost = eva.evaluate(tour)
            gap = (solved_cost - gt_cost) / gt_cost * 100
            # record
            solved_costs[name] = solved_cost
            gt_costs[name] = gt_cost
            gaps[name] = gap 
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
            