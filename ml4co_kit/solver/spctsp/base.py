r"""
Basic solver for Stochastic Prize Collection Traveling Salesman Problem (SPCTSP). 

In the SPCTSP, the expected node prize is known upfront, 
but the real collected prize only becomes known upon visitation.
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


import pickle
import numpy as np
from typing import Union
from ml4co_kit.solver.pctsp.base import PCTSPSolver
from ml4co_kit.evaluate.pctsp import PCTSPEvaluator
from ml4co_kit.utils.time_utils import iterative_execution_for_file
from ml4co_kit.utils.type_utils import to_numpy, TASK_TYPE, SOLVER_TYPE


class SPCTSPSolver(PCTSPSolver):
    r"""
    This class provides a basic framework for solving SPCTSP problems. It includes methods for 
    loading and outputting data in various file formats, normalizing points, and evaluating 
    solutions. Note that the actual solving method should be implemented in subclasses.
    """
    def __init__(
        self, 
        solver_type: SOLVER_TYPE = None, 
        scale: int = 1e6,
        time_limit: float = 60.0,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(SPCTSPSolver, self).__init__(
            solver_type=solver_type, scale=scale, 
            time_limit=time_limit, precision=precision
        )
        self.task_type = TASK_TYPE.SPCTSP
        self.stochastic_norm_prizes: np.ndarray = None
    
    def _check_stochastic_norm_prizes_dim(self):
        r"""
        Ensures that the ``stochastic_norm_prizes`` attribute is a 2D array. 
        If ``stochastic_norm_prizes`` is a 1D array, it adds an additional dimension 
        to make it 2D. Raises a ``ValueError`` if ``stochastic_norm_prizes``
        has more than 2 dimensions.
        """
        if self.stochastic_norm_prizes is not None:
            if self.stochastic_norm_prizes.ndim == 1:
                self.stochastic_norm_prizes = np.expand_dims(self.stochastic_norm_prizes, axis=0)
            if self.stochastic_norm_prizes.ndim != 2:
                raise ValueError("The dimensions of ``stochastic_norm_prizes`` cannot be larger than 2.")
    
    def _check_stochastic_norm_prizes_not_none(self):
        r"""
        Checks if the ``stochastic_norm_prizes`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``norm_prizes`` is ``None``. 
        """
        if self.stochastic_norm_prizes is None:
            message = (
                "``stochastic_norm_prizes`` cannot be None! You can load the ``norm_prizes`` "
                "using the methods including ``from_data``, ``from_txt`` or ``from_pkl``."
            )
            raise ValueError(message)
   
    def _check_constraints_meet(self, ref: bool):
        r"""
        Checks if the ``tour`` satisfies the capacities demands. Raise a `ValueError` if 
        there is a split tour don't meet the demands.
        """
        tours = self.ref_tours if ref else self.tours
        num_tours = len(tours)
        num_instances = self.points.shape[0]
        if num_tours % num_instances != 0:
            raise ValueError(
                "The number of solutions cannot be divided evenly by the number of problems."
            )
        instance_per_sols = num_tours // num_tours
        for idx in range(num_tours):
            cur_prizes = self.stochastic_norm_prizes[idx // instance_per_sols]
            cur_tour = tours[idx]
            cur_collect_prizes = np.sum(cur_prizes[cur_tour[1:-1] - 1])
            if cur_collect_prizes < 1 - 1e-5:
                message = (
                    f"Prize Constraint not met (ref = {ref}) in tour {idx}. "
                    f"The sum of the collected prizes is {cur_collect_prizes}."
                )
                raise ValueError(message)

    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
        show_time: bool = False
    ):
        r"""
        Read data from `.txt` file.
        """
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            # record to lists
            depots_list = list()
            points_list = list()
            penalties_list = list()
            norm_prizes_list = list()
            stochastic_norm_prizes_list = list()
            tours_list = list()
            
            # read by lines
            load_msg = f"Loading data from {file_path}"
            for line in iterative_execution_for_file(file, load_msg, show_time):
                # line to strings
                line = line.strip()
                split_line_0 = line.split("depots ")[1]
                split_line_1 = split_line_0.split(" points ")
                depots = split_line_1[0]
                split_line_2 = split_line_1[1].split(" penalties ")
                points = split_line_2[0]
                split_line_3 = split_line_2[1].split(" norm_prizes ")
                penalties = split_line_3[0]
                split_line_4 = split_line_3[1].split(" stochastic_norm_prizes ")
                norm_prizes = split_line_4[0]
                split_line_5 = split_line_4[1].split(" output ")
                stochastic_norm_prizes = split_line_5[0]
                tours = split_line_5[1]
                
                # strings to array
                depots = depots.split(" ")
                depots = np.array([float(depots[0]), float(depots[1])], dtype=self.precision)
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ], dtype=self.precision
                )
                penalties = penalties.split(" ")
                penalties = np.array(
                    [float(penalties[i]) for i in range(len(penalties))],
                    dtype=self.precision
                )
                norm_prizes = norm_prizes.split(" ")
                norm_prizes = np.array(
                    [float(norm_prizes[i]) for i in range(len(norm_prizes))],
                    dtype=self.precision
                )
                stochastic_norm_prizes = stochastic_norm_prizes.split(" ")
                stochastic_norm_prizes = np.array(
                    [float(stochastic_norm_prizes[i]) for i in range(len(stochastic_norm_prizes))],
                    dtype=self.precision
                )
                tours = tours.split(" ")
                tours = np.array(
                    [int(tours[i]) for i in range(len(tours))]
                )
                
                # add to the list
                depots_list.append(depots)
                points_list.append(points)
                penalties_list.append(penalties)
                norm_prizes_list.append(norm_prizes)
                stochastic_norm_prizes_list.append(stochastic_norm_prizes)
                tours_list.append(tours)

        # check if return list
        if return_list:
            return (
                depots_list, points_list, penalties_list, \
                norm_prizes_list, stochastic_norm_prizes_list, tours_list
            )
            
        depots = np.array(depots_list)
        points = np.array(points_list)
        penalties = np.array(penalties_list)
        norm_prizes = np.array(norm_prizes_list)
        stochastic_norm_prizes = np.array(stochastic_norm_prizes_list)
        tours = tours_list

        # use ``from_data``
        self.from_data(
            depots=depots, points=points, penalties=penalties, norm_prizes=norm_prizes,
            stochastic_norm_prizes=stochastic_norm_prizes, tours=tours, ref=ref, 
            norm=norm, normalize=normalize
        )

    def from_pickle(
        self,
        file_path: str,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
    ):
        # check the file format
        if not file_path.endswith(".pkl"):
            raise ValueError("Invalid file format. Expected a ``.pkl`` file.")
        
        # read the data from .pkl
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # check the data format
        if isinstance(data, list) and len(data) > 0:
            try:
                depots, points, penalties, norm_prizes, stochastic_norm_prizes, tours = zip(*data)
                depots = np.array(depots)
                points = np.array(points)
                penalties = np.array(penalties)
                norm_prizes = np.array(norm_prizes)
                stochastic_norm_prizes = np.array(stochastic_norm_prizes)
                self.from_data(
                    depots=depots, points=points, penalties=penalties, norm_prizes=norm_prizes, 
                    stochastic_norm_prizes=stochastic_norm_prizes, tours=tours, ref=ref, 
                    norm=norm, normalize=normalize
                )
            except Exception as e:
                raise ValueError(f"Invalid data format in PKL file: {e}")
        else:
            raise ValueError("PKL file should contain a list of tuples")

    def from_data(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        penalties: Union[list, np.ndarray] = None,
        norm_prizes: Union[list, np.ndarray] = None,
        stochastic_norm_prizes: Union[list, np.ndarray] = None,
        tours: list = None,
        ref: bool = False,
        norm: str = "EUC_2D",
        normalize: bool = False,
    ):
        """
        Read data from list or np.ndarray.
        """
        # set norm
        self._set_norm(norm)
        
        # depots
        if depots is not None:
            depots = to_numpy(depots)
            self.ori_depots = depots
            self.depots = depots.astype(self.precision)
            self._check_depots_dim()

        # points
        if points is not None:
            points = to_numpy(points)
            self.ori_points = points
            self.points = points.astype(self.precision)
            self._check_ori_points_dim()
        
        # penalties
        if penalties is not None:
            penalties = to_numpy(penalties)
            self.ori_penalties = penalties
            self.penalties = penalties.astype(self.precision)
            self._check_penalties_dim()

        # normalize
        if normalize:
            self._normalize_points_depots_penalties()
    
        # norm_prizes
        if norm_prizes is not None:
            norm_prizes = to_numpy(norm_prizes)
            self.norm_prizes = norm_prizes.astype(self.precision)   
            self._check_norm_prizes_dim()

        # norm_prizes
        if stochastic_norm_prizes is not None:
            stochastic_norm_prizes = to_numpy(stochastic_norm_prizes)
            self.stochastic_norm_prizes = stochastic_norm_prizes.astype(self.precision)   
            self._check_stochastic_norm_prizes_dim()
      
        # tours
        if tours is not None:
            if ref:
                self.ref_tours = tours
                self._check_ref_tours_dim()
            else:
                self.tours = tours
                self._check_tours_dim()

    def to_txt(
        self,
        file_path: str = "example.txt",
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round"
    ):
        r"""
        Output(store) data in ``txt`` format
        """
        # check
        self._check_depots_not_none()
        self._check_points_not_none()
        self._check_penalties_not_none()
        self._check_norm_prizes_not_none()
        self._check_stochastic_norm_prizes_not_none()
        self._check_tours_not_none(ref=False)
        
        # variables
        depots = self.depots
        points = self.ori_points if original else self.points
        penalties = self.penalties
        norm_prizes = self.norm_prizes
        stochastic_norm_prizes = self.stochastic_norm_prizes
        tours = self.tours

        # deal with different shapes and apply scale and dtype
        depots, points, penalties, tours = self._prepare_for_output(
            depots=depots, points=points, penalties=penalties, tours=tours, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )

        # write
        with open(file_path, "w") as f:
            for idx in range(points.shape[0]):
                # write depot
                f.write(f"depots {str(depots[idx][0])} {str(depots[idx][1])} ")

                # write points
                f.write("points ")
                for node_x, node_y in points[idx]:
                    f.write(f"{str(node_x)} {str(node_y)} ")

                # write penalties
                f.write(f"penalties ")
                for i in range(len(penalties[idx])):
                    f.write(f"{str(penalties[idx][i])} ")

                # write norm_prizes
                f.write(f"norm_prizes ")
                for i in range(len(norm_prizes[idx])):
                    f.write(f"{str(norm_prizes[idx][i])} ")

                # write norm_prizes
                f.write(f"stochastic_norm_prizes ")
                for i in range(len(stochastic_norm_prizes[idx])):
                    f.write(f"{str(stochastic_norm_prizes[idx][i])} ")
          
                # write tours
                f.write(f"output ")
                for node_idx in tours[idx]:
                    f.write(f"{node_idx} ")
                f.write("\n")
            f.close()
    
    def to_pickle(
        self,
        file_path: str = "example.pkl",
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round"
    ):
        r"""
        Output(store) data in ``pkl`` format
        """
        # check
        self._check_depots_not_none()
        self._check_points_not_none()
        self._check_penalties_not_none()
        self._check_norm_prizes_not_none()
        self._check_tours_not_none(ref=False)
        
        # variables
        depots = self.ori_depots if original else self.depots
        points = self.ori_points if original else self.points
        penalties = self.ori_penalties if original else self.penalties
        norm_prizes = self.norm_prizes
        stochastic_norm_prizes = self.stochastic_norm_prizes
        tours = self.tours

        # deal with different shapes and apply scale and dtype
        depots, points, penalties, tours = self._prepare_for_output(
            depots=depots, points=points, penalties=penalties, tours=tours, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )

        # write
        with open(file_path, "wb") as f:
            pickle.dump(
                list(zip(
                    depots, points, penalties, norm_prizes, 
                    stochastic_norm_prizes, tours
                )), f, pickle.HIGHEST_PROTOCOL
            )
    
    def evaluate(
        self,
        calculate_gap: bool = False,
        check_constraints: bool = True,
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round",
    ):
        """
        Evaluate the solution quality of the solver
        """
        # check
        self._check_depots_not_none()
        self._check_points_not_none()
        self._check_penalties_not_none()
        self._check_norm_prizes_not_none()
        self._check_tours_not_none(ref=False)
        if check_constraints:
            self._check_constraints_meet(ref=False)
        if calculate_gap:
            self._check_tours_not_none(ref=True)
            if check_constraints:
                self._check_constraints_meet(ref=True)
            
        # variables
        depots = self.ori_depots if original else self.points
        points = self.ori_points if original else self.points
        penalties = self.ori_penalties if original else self.penalties
        tours = self.tours
        ref_tours = self.ref_tours

        # apply scale and dtype
        depots, points, penalties = self._apply_scale_and_dtype(
            depots=depots, points=points, penalties=penalties, 
            apply_scale=apply_scale, to_int=to_int, round_func=round_func
        )

        # prepare for evaluate
        tours_cost_list = list()
        samples = points.shape[0]
        if calculate_gap:
            ref_tours_cost_list = list()
            gap_list = list()

        # deal with different situation
        if len(tours) != samples:
            if len(tours) % samples != 0:
                raise ValueError("The number of solutions cannot be divided evenly by the number of problems.")
            per_tour_num = len(tours) // samples
            for idx in range(samples):
                evaluator = PCTSPEvaluator(
                    depots=depots[idx], 
                    points=points[idx], 
                    penalties=penalties[idx],
                    norm=self.norm
                )
                solved_tours = tours[idx*per_tour_num, (idx+1)*per_tour_num]
                solved_costs = list()
                for tour in solved_tours:
                    solved_costs.append(evaluator.evaluate(
                        route=tour, to_int=to_int, round_func=round_func
                    ))
                solved_cost = np.min(solved_costs)
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(
                        route=ref_tours[idx], to_int=to_int, round_func=round_func
                    )
                    ref_tours_cost_list.append(ref_cost)
                    gap = (solved_cost - ref_cost) / ref_cost * 100
                    gap_list.append(gap)
        else:
            # a problem only one solved tour
            for idx in range(samples):
                evaluator = PCTSPEvaluator(
                    depots=depots[idx], 
                    points=points[idx], 
                    penalties=penalties[idx],
                    norm=self.norm
                )
                solved_cost = evaluator.evaluate(
                    route=tours[idx], to_int=to_int, round_func=round_func
                )
                tours_cost_list.append(solved_cost)
                if calculate_gap:
                    ref_cost = evaluator.evaluate(
                        route=ref_tours[idx], to_int=to_int, round_func=round_func
                    )
                    ref_tours_cost_list.append(ref_cost)
                    gap = (solved_cost - ref_cost) / ref_cost * 100
                    gap_list.append(gap)

        # calculate average cost/gap & std
        tours_costs = np.array(tours_cost_list)
        if calculate_gap:
            ref_costs = np.array(ref_tours_cost_list)
            gaps = np.array(gap_list)
        costs_avg = np.average(tours_costs)
        if calculate_gap:
            ref_costs_avg = np.average(ref_costs)
            gap_avg = np.sum(gaps) / samples
            gap_std = np.std(gaps)
            return costs_avg, ref_costs_avg, gap_avg, gap_std
        else:
            return costs_avg
        
    def solve(
        self,
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        penalties: Union[list, np.ndarray] = None,
        norm_prizes: Union[list, np.ndarray] = None,
        stochastic_norm_prizes: Union[list, np.ndarray] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        This method will be implemented in subclasses.
        """
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "SPCTSPSolver"