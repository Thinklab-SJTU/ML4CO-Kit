import numpy as np
import data4co.utils.tsp_utils as tsp_utils
from typing import Union
from data4co.utils.tsp_utils import TSPEvaluator


class TSPSolver:
    def __init__(self):
        self.solver_type = None
        self.points = None
        self.ori_points = None
        self.tours = None
        self.gt_tours = None
        self.nodes_num = None

    def check_points_dim(self):
        if self.points is None:
            return
        elif self.points.ndim == 2:
            self.points = np.expand_dims(self.points, axis=0)
        self.nodes_num = self.points.shape[1]

    def from_tspfile(self, filename: str):
        if not filename.endswith(".tsp"):
            raise ValueError("Invalid file format. Expected a .tsp file.")
        points, _ = tsp_utils.get_data_from_tsp_file(filename)
        if points is None:
            raise RuntimeError("Error in loading {}".format(filename))
        self.ori_points = points
        self.points = points.astype(np.float32)
        self.check_points_dim()

    def from_txt(self, filename: str, load_gt_tours: str=True):
        if not filename.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a .txt file.")
        with open(filename, 'r') as file:
            nodes_coords = list()
            gt_tours = list()
            for line in file:
                line = line.strip()
                if 'output' in line:
                    split_line = line.split(' output ')
                    points = split_line[0]
                    tour = split_line[1]
                    tour = tour.split(' ')
                    tour = np.array([int(t) for t in tour])
                    tour -= 1
                    gt_tours.append(tour)
                else:
                    points = line
                    load_gt_tours = False
                points = points.split(' ')
                points = np.array([
                    [float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)
                ])
                nodes_coords.append(points)
        
        if load_gt_tours:
            self.gt_tours = np.array(gt_tours)
        nodes_coords = np.array(nodes_coords)
        self.ori_points = nodes_coords
        self.points = nodes_coords.astype(np.float32)
        self.check_points_dim()

    def from_data(self, points: np.ndarray):
        self.points = points.astype(np.float32)
        self.check_points_dim()

    def read_gt_tours(self, gt_tours: np.ndarray):
        if gt_tours.ndim == 1:
            gt_tours = np.expand_dims(gt_tours, axis=0)
        self.gt_tours = gt_tours
   
    def read_gt_tours_from_tspfile(self, filename: str):
        if not filename.endswith(".opt.tour"):
            raise ValueError("Invalid file format. Expected a .opt.tour file.")
        gt_tour = tsp_utils.get_tour_from_tour_file(filename)
        self.gt_tours = np.expand_dims(gt_tour, axis=0)
    
    def to_tsp_file(
        self,
        tour_filename: str=None,
        filename: str=None,
        points: Union[np.ndarray, list]=None,
        tours: Union[np.ndarray, list]=None,
    ):
        # points
        if points is None:
            points = self.points
        if type(points) == list:
            points = np.array(points)
        # tours
        if tours is None:
            tours = self.tours
        if type(tours) == list:
            tours = np.array(tours)
        # generate .tsp file if filename is not none
        if filename is not None:
            assert filename.endswith('.tsp')
            tsp_utils.generate_tsp_file(points, filename)
        # generate .tsp.tour file if tour_filename is not none
        if tour_filename is not None:
            assert filename.endswith('.tsp.tour')
            tsp_utils.generate_opt_tour_file(tours, filename)
            
    def to_txt(
        self, 
        filename: str="example.txt",
        points: Union[np.ndarray, list]=None,
        tours: Union[np.ndarray, list]=None,
    ):
        # points
        if points is None:
            points = self.ori_points
        if type(points) == list:
            points = np.array(points)
        # tours
        if tours is None:
            tours = self.tours
        if type(tours) == list:
            tours = np.array(tours)
        # write
        with open(filename, "w") as f:
            for idx, tour in enumerate(tours):
                f.write(" ".join(str(x) + str(" ") + str(y) for x, y in points[idx]))
                f.write(str(" ") + str('output') + str(" "))
                f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                f.write("\n")
            f.close()

    def evaluate(self, caculate_gap: bool=False):
        """
        """
        if caculate_gap and self.gt_tours is None:
            raise ValueError("gt_tours cannot be None, please use TSPLKHSolver to get the gt_tours.")
        if self.tours is None:
            raise ValueError("tours cannot be None, please use method 'solve' to get solution.")
        
        if self.points.ndim == 2:
            # only one problem
            evaluator = TSPEvaluator(self.points)
            solved_cost = evaluator.evaluate(self.tours)
            return solved_cost
        else:
            # more than one problem
            cost_total = 0
            samples = self.points.shape[0]
            if caculate_gap:
                gap_list = list()
            if self.tours.shape[0] != samples:
                # a problem has more than one solved tour
                tours = self.tours.reshape(samples, -1, self.tours.shape[-1])
                for idx in range(samples):
                    evaluator = TSPEvaluator(self.points[idx])
                    solved_tours = tours[idx]
                    solved_costs = list()
                    for tour in solved_tours:
                        solved_costs.append(evaluator.evaluate(tour))
                    solved_cost = np.min(solved_costs)
                    cost_total += solved_cost
                    if caculate_gap:
                        gt_cost = evaluator.evaluate(self.gt_tours[idx])
                        gap = (solved_cost - gt_cost) / gt_cost * 100
                        gap_list.append(gap)
            else:
                # a problem only one solved tour
                for idx in range(samples):
                    evaluator = TSPEvaluator(self.points[idx])
                    solved_tour = self.tours[idx]
                    solved_cost = evaluator.evaluate(solved_tour)
                    cost_total += solved_cost
                    if caculate_gap:
                        gt_cost = evaluator.evaluate(self.gt_tours[idx])
                        gap = (solved_cost - gt_cost) / gt_cost * 100
                        gap_list.append(gap)
            # caculate average cost/gap & std
            cost_avg = cost_total / samples
            if caculate_gap:
                gap_avg = np.sum(gap_list) / samples
                gap_std = np.std(gap_list)
                return cost_avg, gap_avg, gap_std
            else:
                return cost_avg

    def solve(self, batch_size: int=1, points: np.ndarray=None) -> np.ndarray:
        raise NotImplementedError("solve is required to implemented in subclass")
