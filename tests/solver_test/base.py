r"""
Base class for solver testers.
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


import pathlib
from typing import Type, List
from ml4co_kit import SolverBase, TaskBase, TASK_TYPE
from ml4co_kit import (
    TSPTask, ATSPTask, CVRPTask, OPTask, PCTSPTask, SPCTSPTask,
    MClTask, MCutTask, MISTask, MVCTask
)


class SolverTesterBase(object):
    def __init__(
        self, 
        test_solver_class: Type[SolverBase],
        test_task_type_list: List[TASK_TYPE],
        test_args_list: List[dict],
        exclude_test_files_list: List[List[pathlib.Path]]
    ):
        self.test_solver_class = test_solver_class
        self.test_task_type_list = test_task_type_list
        self.test_args_list = test_args_list
        self.exclude_test_files_list = exclude_test_files_list

    def pre_test(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def test(self):
        # Things to do before test
        self.pre_test()
        
        # Test for each distribution type
        print(f"\nTesting {str(self.test_solver_class.__name__)}")
        for test_task_type, test_args, exclude_test_files in zip(
            self.test_task_type_list, self.test_args_list, self.exclude_test_files_list
        ):
            try:
                test_task_list = self.get_task_list(test_task_type, exclude_test_files)
                solver = self.test_solver_class(**test_args)
                for test_task in test_task_list:
                    solver.solve(test_task)
                    eval_results = test_task.evaluate_w_gap()
                    print(f"{str(test_task)} Eval results: {eval_results}")
            except Exception as e:
                raise ValueError(
                    f"Error ``{e}`` occurred when testing {self.test_solver_class.__name__}\n"
                    f"Test args: {test_args}, Task: {test_task} "
                )
    
    def get_task_list(
        self, 
        test_task_type: TASK_TYPE, 
        exclude_test_files: List[pathlib.Path]
    ) -> List[TaskBase]:
        
        # Routing Problems
        if test_task_type == TASK_TYPE.ATSP:
            return self._get_atsp_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRP:
            return self._get_cvrp_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.OP:
            return self._get_op_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.PCTSP:
            return self._get_pctsp_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.SPCTSP:
            return self._get_spctsp_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.TSP:
            return self._get_tsp_tasks(exclude_test_files)
        
        # Graph Problems
        elif test_task_type == TASK_TYPE.MCL:
            return self._get_mcl_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.MCUT:
            return self._get_mcut_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.MIS:
            return self._get_mis_tasks(exclude_test_files)
        elif test_task_type == TASK_TYPE.MVC:
            return self._get_mvc_tasks(exclude_test_files)
        
        # Others
        else:
            raise ValueError(
                f"Test task type {test_task_type} is not supported."
            )
    
    ########################################
    #           Routing Problems           #
    ########################################
    
    def _get_atsp_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[ATSPTask]:
        atsp_test_files_list = [
            pathlib.Path("test_dataset/atsp/task/atsp50_hcp_task.pkl"),
            pathlib.Path("test_dataset/atsp/task/atsp50_uniform_task.pkl"),
            pathlib.Path("test_dataset/atsp/task/atsp54_sat_task.pkl"),
            pathlib.Path("test_dataset/atsp/task/atsp500_uniform_task.pkl"),
        ]
        task_list = list()
        for test_file in atsp_test_files_list:
            if test_file not in exclude_test_files:
                task = ATSPTask()
                task.from_pickle(test_file)
                task_list.append(task)
        return task_list

    def _get_cvrp_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[CVRPTask]:
        cvrp_test_files_list = [
            pathlib.Path("test_dataset/cvrp/task/cvrp50_uniform_task.pkl"),
            pathlib.Path("test_dataset/cvrp/task/cvrp500_uniform_task.pkl"),
        ]
        task_list = list()
        for test_file in cvrp_test_files_list:
            if test_file not in exclude_test_files:
                task = CVRPTask()
                task.from_pickle(test_file)
                task_list.append(task)
        return task_list
    
    def _get_op_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[OPTask]:
        pass
    
    def _get_pctsp_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[PCTSPTask]:
        pass
    
    def _get_spctsp_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[SPCTSPTask]:
        pass
    
    def _get_tsp_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[TSPTask]:
        tsp_test_files_list_1 = [
            pathlib.Path("test_dataset/tsp/task/tsp50_cluster_task.pkl"),
            pathlib.Path("test_dataset/tsp/task/tsp50_gaussian_task.pkl"),
        ]
        tsp_test_files_list_2 = [
            pathlib.Path("test_dataset/tsp/task/tsp50_uniform_task.pkl"),
            pathlib.Path("test_dataset/tsp/task/tsp500_uniform_task.pkl"),
        ]
        task_list = list()
        for test_file in tsp_test_files_list_1:
            if test_file not in exclude_test_files:
                task = TSPTask()
                task.from_pickle(test_file)
                task._normalize_points()
                task_list.append(task)
        for test_file in tsp_test_files_list_2:
            if test_file not in exclude_test_files:
                task = TSPTask()
                task.from_pickle(test_file)
                task_list.append(task)
        return task_list

    ########################################
    #            Graph Problems            #
    ########################################
      
    def _get_mcl_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[MClTask]:
        mcl_test_files_list = [
            pathlib.Path("test_dataset/mcl/task/mcl_rb-large_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mcl/task/mcl_rb-small_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mcl/task/mcl_rb-small_uniform-weighted_task.pkl")
        ]
        task_list = list()
        for test_file in mcl_test_files_list:
            if test_file not in exclude_test_files:
                task = MClTask()
                task.from_pickle(test_file)
                task_list.append(task)
        return task_list
    
    def _get_mcut_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[MCutTask]:
        mcut_test_files_list = [
            pathlib.Path("test_dataset/mcut/task/mcut_ba-large_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mcut/task/mcut_ba-small_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mcut/task/mcut_ba-small_uniform-weighted_task.pkl")
        ]
        task_list = list()
        for test_file in mcut_test_files_list:
            if test_file not in exclude_test_files:
                task = MCutTask()
                task.from_pickle(test_file)
                task_list.append(task)
        return task_list
    
    def _get_mis_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[MISTask]:
        mis_test_files_list = [
            pathlib.Path("test_dataset/mis/task/mis_er-700-800_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mis/task/mis_rb-small_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mis/task/mis_rb-small_uniform-weighted_task.pkl"),
            pathlib.Path("test_dataset/mis/task/mis_satlib_no-weighted_task.pkl")
        ]
        task_list = list()
        for test_file in mis_test_files_list:
            if test_file not in exclude_test_files:
                task = MISTask()
                task.from_pickle(test_file)
                task_list.append(task)
        return task_list
    
    def _get_mvc_tasks(self, exclude_test_files: List[pathlib.Path]) -> List[MVCTask]:
        mvc_test_files_list = [
            pathlib.Path("test_dataset/mvc/task/mvc_rb-large_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mvc/task/mvc_rb-small_no-weighted_task.pkl"),
            pathlib.Path("test_dataset/mvc/task/mvc_rb-small_uniform-weighted_task.pkl"),
        ]
        task_list = list()
        for test_file in mvc_test_files_list:
            if test_file not in exclude_test_files:
                task = MVCTask()
                task.from_pickle(test_file)
                task_list.append(task)
        return task_list