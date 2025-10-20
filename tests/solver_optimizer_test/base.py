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
import re
from typing import Type, List
from ml4co_kit import SolverBase, TaskBase, TASK_TYPE
from ml4co_kit import (
    TSPTask, ATSPTask, CVRPTask, OPTask, PCTSPTask, SPCTSPTask,
    MClTask, MCutTask, MISTask, MVCTask,
    MinVarPOTask, MaxRetPOTask, MOPOTask
)
from ml4co_kit import (
    TSPWrapper, ATSPWrapper, CVRPWrapper, OPWrapper, PCTSPWrapper, SPCTSPWrapper,
    MClWrapper, MCutWrapper, MISWrapper, MVCWrapper,
    MinVarPOWrapper, MaxRetPOWrapper, MOPOWrapper
)


class SolverTesterBase(object):
    def __init__(
        self, 
        mode_list: List[str],
        test_solver_class: Type[SolverBase],
        test_task_type_list: List[TASK_TYPE],
        test_args_list: List[dict],
        exclude_test_files_list: List[List[pathlib.Path]]
    ):
        self.mode_list = mode_list
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
                for mode in self.mode_list:
                    solver = self.test_solver_class(**test_args)
                    if mode == "solve":
                        test_task_list = self.get_task_list(
                            mode=mode, 
                            test_task_type=test_task_type, 
                            exclude_test_files=exclude_test_files
                        )
                        for test_task in test_task_list:
                            solver.solve(test_task)
                            eval_results = test_task.evaluate_w_gap()
                            print(f"{str(test_task)} Eval results: {eval_results}")
                    if mode == "batch_solve":
                        batch_test_task_list = self.get_task_list(
                            mode=mode, 
                            test_task_type=test_task_type, 
                            exclude_test_files=exclude_test_files
                        )
                        for batch_test_task in batch_test_task_list:
                            solver.batch_solve(batch_test_task)
                            for test_task in batch_test_task:
                                test_task: TaskBase
                                eval_results = test_task.evaluate_w_gap()
                                print(f"{str(test_task)} Eval results: {eval_results}")          
            except Exception as e:
                raise ValueError(
                    f"Error ``{e}`` occurred when testing {self.test_solver_class.__name__}\n"
                    f"Test args: {test_args}, Mode: {mode}, Task: {test_task_type} "
                )
    
    def get_task_list(
        self, 
        mode: str,
        test_task_type: TASK_TYPE, 
        exclude_test_files: List[pathlib.Path]
    ) -> List[TaskBase]:
        
        # Routing Problems
        if test_task_type == TASK_TYPE.ATSP:
            return self._get_atsp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRP:
            return self._get_cvrp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.OP:
            return self._get_op_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.PCTSP:
            return self._get_pctsp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.SPCTSP:
            return self._get_spctsp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.TSP:
            return self._get_tsp_tasks(mode, exclude_test_files)
        
        # Graph Problems
        elif test_task_type == TASK_TYPE.MCL:
            return self._get_mcl_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MCUT:
            return self._get_mcut_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MIS:
            return self._get_mis_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MVC:
            return self._get_mvc_tasks(mode, exclude_test_files)
        
        # Portfolio Problems
        elif test_task_type == TASK_TYPE.MINVARPO:
            return self._get_minvarpo_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MAXRETPO:
            return self._get_maxretpo_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MOPO:
            return self._get_mopo_tasks(mode, exclude_test_files)

        # Others
        else:
            raise ValueError(
                f"Test task type {test_task_type} is not supported."
            )
    
    ########################################
    #           Routing Problems           #
    ########################################
    
    def _get_atsp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[ATSPTask]:
        # ``Solve`` mode
        if mode == "solve":
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
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            atsp_test_files_list = [
                pathlib.Path("test_dataset/atsp/wrapper/atsp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/atsp/wrapper/atsp500_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in atsp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = ATSPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
        
    def _get_cvrp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPTask]:
        # ``Solve`` mode
        if mode == "solve":
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
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrp_test_files_list = [
                pathlib.Path("test_dataset/tsp/wrapper/tsp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/tsp/wrapper/tsp500_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_op_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[OPTask]:
        # ``Solve`` mode
        if mode == "solve":
            op_test_files_list = [
                pathlib.Path("test_dataset/op/task/op50_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in op_test_files_list:
                if test_file not in exclude_test_files:
                    task = OPTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            op_test_files_list = [
                pathlib.Path("test_dataset/op/wrapper/op50_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in op_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = OPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
    
    def _get_pctsp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[PCTSPTask]:
        # ``Solve`` mode
        if mode == "solve":
            pctsp_test_files_list = [
                pathlib.Path("test_dataset/pctsp/task/pctsp50_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in pctsp_test_files_list:
                if test_file not in exclude_test_files:
                    task = PCTSPTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            pctsp_test_files_list = [
                pathlib.Path("test_dataset/pctsp/wrapper/pctsp50_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in pctsp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = PCTSPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_spctsp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[SPCTSPTask]:
        # ``Solve`` mode
        if mode == "solve":
            spctsp_test_files_list = [
                pathlib.Path("test_dataset/spctsp/task/spctsp50_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in spctsp_test_files_list:
                if test_file not in exclude_test_files:
                    task = SPCTSPTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            spctsp_test_files_list = [
                pathlib.Path("test_dataset/spctsp/wrapper/spctsp50_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in spctsp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = SPCTSPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
    
    def _get_tsp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[TSPTask]:
        # ``Solve`` mode
        if mode == "solve":
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
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            tsp_test_files_list = [
                pathlib.Path("test_dataset/tsp/wrapper/tsp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/tsp/wrapper/tsp500_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in tsp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = TSPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
        
    ########################################
    #            Graph Problems            #
    ########################################
      
    def _get_mcl_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MClTask]:
        # ``Solve`` mode
        if mode == "solve":
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

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            mcl_test_files_list = [
                pathlib.Path("test_dataset/mcl/task/mcl_rb-large_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/mcl/task/mcl_rb-small_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/mcl/task/mcl_rb-small_uniform-weighted_task.pkl"),
            ]
            bacth_task_list = list()
            for test_file in mcl_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MClWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_mcut_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MCutTask]:
        # ``Solve`` mode
        if mode == "solve":
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

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            mcut_test_files_list = [
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-large_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mcut/wrapper/mcut_ba-small_uniform-weighted_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in mcut_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MCutWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
        
    def _get_mis_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MISTask]:
        # ``Solve`` mode
        if mode == "solve":
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

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            mis_test_files_list = [
                pathlib.Path("test_dataset/mis/wrapper/mis_er-700-800_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mis/wrapper/mis_satlib_no-weighted_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in mis_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MISWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
        
    def _get_mvc_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MVCTask]:
        # ``Solve`` mode
        if mode == "solve":
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
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            mvc_test_files_list = [
                pathlib.Path("test_dataset/mvc/wrapper/mvc_rb-large_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mvc/wrapper/mvc_rb-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/mvc/wrapper/mvc_rb-small_uniform-weighted_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in mvc_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MVCWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    ########################################
    #         Portfolio Problems           #
    ########################################
    
    def _get_minvarpo_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MinVarPOTask]:
        # ``Solve`` mode
        if mode == "solve":
            minvarpo_test_files_list = [
                pathlib.Path("test_dataset/minvarpo/task/minvarpo_gbm_task.pkl"),
            ]
            task_list = list()
            for test_file in minvarpo_test_files_list:
                if test_file not in exclude_test_files:
                    task = MinVarPOTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            minvarpo_test_files_list = [
                pathlib.Path("test_dataset/minvarpo/wrapper/minvarpo_gbm_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in minvarpo_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MinVarPOWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_maxretpo_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MaxRetPOTask]:
        # ``Solve`` mode
        if mode == "solve":
            maxretpo_test_files_list = [
                pathlib.Path("test_dataset/maxretpo/task/maxretpo_gbm_task.pkl"),
            ]
            task_list = list()
            for test_file in maxretpo_test_files_list:
                if test_file not in exclude_test_files:
                    task = MaxRetPOTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            maxretpo_test_files_list = [
                pathlib.Path("test_dataset/maxretpo/wrapper/maxretpo_gbm_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in maxretpo_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MaxRetPOWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_mopo_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MOPOTask]:
        # ``Solve`` mode
        if mode == "solve":
            mopo_test_files_list = [
                pathlib.Path("test_dataset/mopo/task/mopo_gbm_task.pkl"),
            ]
            task_list = list()
            for test_file in mopo_test_files_list:
                if test_file not in exclude_test_files:
                    task = MOPOTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            mopo_test_files_list = [
                pathlib.Path("test_dataset/mopo/wrapper/mopo_gbm_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in mopo_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MOPOWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list