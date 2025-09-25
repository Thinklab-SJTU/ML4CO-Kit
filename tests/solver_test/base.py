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
from tqdm import tqdm
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
        test_files_list: List[pathlib.Path],
        test_tasks_list: List[TASK_TYPE],
        test_args_list: List[dict]
    ):
        self.test_solver_class = test_solver_class
        self.test_files_list = test_files_list
        self.test_tasks_list = test_tasks_list
        self.test_args_list = test_args_list

    def pre_test(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def test(self):
        # Things to do before test
        self.pre_test()
        
        # Test for each distribution type
        print(f"Testing {str(self.test_solver_class.__name__)}")
        for test_task_type, test_file, test_args in zip(
            self.test_tasks_list, self.test_files_list, self.test_args_list
        ):
            try:
                test_instance = self.get_instance(test_task_type, test_file)
                solver = self.test_solver_class(**test_args)
                solver.solve(test_instance)
                eval_results = test_instance.evaluate_w_gap()
                print(f"{str(test_instance)} Eval results: {eval_results}")
            except:
                raise ValueError(
                    f"Error occurred when testing {self.test_solver_class.__name__}\n"
                    f"Test args: {test_args} "
                )
    
    def get_instance(self, test_task_type: TASK_TYPE, test_file: pathlib.Path) -> TaskBase:
        # Initialize Task
        if test_task_type == TASK_TYPE.TSP:
            task = TSPTask()
        elif test_task_type == TASK_TYPE.ATSP:
            task = ATSPTask()
        elif test_task_type == TASK_TYPE.CVRP:
            task = CVRPTask()
        elif test_task_type == TASK_TYPE.OP:
            task = OPTask()
        elif test_task_type == TASK_TYPE.PCTSP:
            task = PCTSPTask()
        elif test_task_type == TASK_TYPE.SPCTSP:
            task = SPCTSPTask()
        elif test_task_type == TASK_TYPE.MCL:
            task = MClTask()
        elif test_task_type == TASK_TYPE.MCUT:
            task = MCutTask()
        elif test_task_type == TASK_TYPE.MIS:
            task = MISTask()
        elif test_task_type == TASK_TYPE.MVC:
            task = MVCTask()
        else:
            raise ValueError(
                f"Test task type {test_task_type} is not supported."
            )
        
        # Load Task from pickle file
        task.from_pickle(test_file)
        return task