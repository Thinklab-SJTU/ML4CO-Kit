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

import numpy as np
from ml4co_kit.solver.pctsp.base import PCTSPSolver
from ml4co_kit.utils.type_utils import TASK_TYPE, SOLVER_TYPE


class SPCTSPSolver(PCTSPSolver):
    r"""
    This class provides a basic framework for solving SPCTSP problems. It includes methods for 
    loading and outputting data in various file formats, normalizing points, and evaluating 
    solutions. Note that the actual solving method should be implemented in subclasses.
    """
    def __init__(
        self, 
        solver_type: SOLVER_TYPE = None, 
        scale: int = 1e7,
        time_limit: float = 60.0
    ):
        super(SPCTSPSolver, self).__init__(
            solver_type=solver_type,
            scale=scale,
            time_limit=time_limit
        )
        self.task_type = TASK_TYPE.SPCTSP
        
    def evaluate(
        self,
        calculate_gap: bool = False,
        original: bool = True,
        apply_scale: bool = False,
        to_int: bool = False,
        round_func: str = "round",
    ):
        """
        Evaluate the solution quality of the solver

        :param calculate_gap: boolean, whether to calculate the gap with the reference solutions.
        :param original: boolean, whether to use ``original points`` or ``points``.
        :param apply_scale: boolean, whether to perform data scaling for the corrdinates.
        :param to_int: boolean, whether to transfer the corrdinates to integters.
        :param round_func: string, the category of the rounding function, used when ``to_int`` is True.

        .. note::
            - Please make sure the ``points`` and the ``tours`` are not None.
            - If you set the ``calculate_gap`` as True, please make sure the ``ref_tours`` is not None.
        
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import SPCTSPRepotSolver
                
                # create SPCTSPRepotSolver
                >>> solver = SPCTSPRepotSolver()

                # load data and reference solutions from ``.txt`` file
                >>> solver.from_txt(file_path="examples/pctsp/txt/pctsp50.txt")
                
                # solve
                >>> solver.solve()
                    
                # Evaluate the quality of the solutions solved by REOPT
                >>> solver.evaluate(calculate_gap=False)
        """
        # check
        self._check_depots_not_none()
        if original:
            self._check_ori_points_not_none()
        else:
            self._check_points_not_none()
        self._check_penalties_not_none()
        self._check_stochastic_prizes_not_none()
        self._check_tours_not_none(ref=False)
        if calculate_gap:
            self._check_tours_not_none(ref=True)
            
        # variables
        depots = self.depots
        points = self.ori_points if original else self.points
        penalties = self.penalties
        stochastic_prizes = self.stochastic_prizes
        tours = self.tours
        ref_tours = self.ref_tours

        # apply scale and dtype
        points = self._apply_scale_and_dtype(
            points=points, apply_scale=apply_scale,
            to_int=to_int, round_func=round_func
        )

        # prepare for evaluate
        tours_cost_list = list()
        samples = points.shape[0]
        if calculate_gap:
            ref_tours_cost_list = list()
            gap_list = list()

        if len(tours) != samples:
            raise NotImplementedError(
                "Evaluation is not implemented for multiple tours per instance."
            )
        
        # Suppose a problem only have one solved tour
        for idx in range(samples):
            solved_cost = self.calc_pctsp_cost(
                depot=depots[idx],
                loc=points[idx],
                penalty=penalties[idx],
                prize=stochastic_prizes[idx],
                tour=tours[idx]
            )
            tours_cost_list.append(solved_cost)
            if calculate_gap:
                ref_cost = self.calc_pctsp_cost(
                    depot=depots[idx],
                    loc=points[idx],
                    penalty=penalties[idx],
                    prize=stochastic_prizes[idx],
                    tour=ref_tours[idx]
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