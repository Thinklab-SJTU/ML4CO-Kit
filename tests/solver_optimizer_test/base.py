r"""
Base class for solver and optimizer testers.
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
from ml4co_kit import *


class SolverTesterBase(object):
    def __init__(
        self, 
        mode_list: List[str],
        test_solver_class: Type[SolverBase],
        test_task_type_list: List[TASK_TYPE],
        test_args_list: List[dict],
        exclude_test_files_list: List[List[pathlib.Path]],
        info: str = None
    ):
        self.info = info
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
        
        # Print test information
        print(f"\n--------------------------------------------------")
        if self.info is not None:
            print(f"Testing {str(self.test_solver_class.__name__)} ({self.info})")
        else:
            print(f"Testing {str(self.test_solver_class.__name__)}")

        # Test for each distribution type
        for test_task_type, test_args, exclude_test_files in zip(
            self.test_task_type_list, self.test_args_list, self.exclude_test_files_list
        ):
            # Print test task type
            print(f"\nTest task type: {test_task_type}")
            if "optimizer" in test_args:
                print(f"Optimizer: {test_args['optimizer'].__class__.__name__}")

            # Test for each mode
            try:
                for mode in self.mode_list:
                    # Initialize solver
                    solver = self.test_solver_class(**test_args)
                    print(f"Mode = {mode}")
                    
                    # Solve mode
                    if mode == "solve":
                        # Get task list
                        test_task_list = self.get_task_list(
                            mode=mode, 
                            test_task_type=test_task_type, 
                            exclude_test_files=exclude_test_files
                        )

                        # Solve tasks
                        for test_task in test_task_list:
                            solver.solve(test_task)
                            if test_task.task_type not in [TASK_TYPE.EDAP]:
                                eval_results = test_task.evaluate_w_gap()
                                print(f"{str(test_task)} Eval results: {eval_results}")
                            else:
                                eval_sol_results = test_task.evaluate(test_task.sol)
                                eval_ref_results = test_task.evaluate(test_task.ref_sol)
                                print(f"{str(test_task)} Eval Sol results: {eval_sol_results}")
                                print(f"{str(test_task)} Eval Ref results: {eval_ref_results}")
                        del test_task_list

                    # Batch solve mode
                    if "batch_solve" in mode:
                        batch_test_task_list = self.get_task_list(
                            mode="batch_solve", 
                            test_task_type=test_task_type, 
                            exclude_test_files=exclude_test_files
                        )
                        optimizer_parallel = True if "parallel" in mode else False
                        for batch_test_task in batch_test_task_list:
                            solver.batch_solve(batch_test_task, optimizer_parallel=optimizer_parallel)
                            for test_task in batch_test_task:
                                test_task: TaskBase
                                eval_results = test_task.evaluate_w_gap()
                                print(f"{str(test_task)} Eval results: {eval_results}")
                            del batch_test_task

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
        elif test_task_type == TASK_TYPE.OP:
            return self._get_op_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.PCTSP:
            return self._get_pctsp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.SPCTSP:
            return self._get_spctsp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.TSP:
            return self._get_tsp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRP:
            return self._get_cvrp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRPB:
            return self._get_cvrpb_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRPBL:
            return self._get_cvrpbl_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRPBLTW:
            return self._get_cvrpbltw_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRPBTW:
            return self._get_cvrpbtw_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRPL:
            return self._get_cvrpl_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRPLTW:
            return self._get_cvrpltw_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.CVRPTW:
            return self._get_cvrptw_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MTVRP:
            return self._get_mtvrp_tasks(mode, exclude_test_files)

        # Graph Problems
        elif test_task_type == TASK_TYPE.MCL:
            return self._get_mcl_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MCUT:
            return self._get_mcut_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MIS:
            return self._get_mis_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MVC:
            return self._get_mvc_tasks(mode, exclude_test_files)
        
        # Graph Set Problems
        elif test_task_type == TASK_TYPE.GM:
            return self._get_gm_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.GED:
            return self._get_ged_tasks(mode, exclude_test_files)
        
        # Portfolio Problems
        elif test_task_type == TASK_TYPE.MINVARPO:
            return self._get_minvarpo_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MAXRETPO:
            return self._get_maxretpo_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.MOPO:
            return self._get_mopo_tasks(mode, exclude_test_files)

        # SAT Problems
        elif test_task_type == TASK_TYPE.SATP:
            return self._get_satp_tasks(mode, exclude_test_files)
        elif test_task_type == TASK_TYPE.SATA:
            return self._get_sata_tasks(mode, exclude_test_files)
        
        # EDA Problems
        elif test_task_type == TASK_TYPE.EDAP:
            return self._get_edap_tasks(mode, exclude_test_files)
        
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
                pathlib.Path("test_dataset/routing/tsp/atsp/task/atsp50_hcp_task.pkl"),
                pathlib.Path("test_dataset/routing/tsp/atsp/task/atsp50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/tsp/atsp/task/atsp54_sat_task.pkl"),
                pathlib.Path("test_dataset/routing/tsp/atsp/task/atsp500_uniform_task.pkl"),
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
                pathlib.Path("test_dataset/routing/tsp/atsp/wrapper/atsp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/tsp/atsp/wrapper/atsp500_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in atsp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = ATSPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_op_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[OPTask]:
        # ``Solve`` mode
        if mode == "solve":
            op_test_files_list = [
                pathlib.Path("test_dataset/routing/tsp/op/task/op50_uniform_task.pkl"),
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
                pathlib.Path("test_dataset/routing/tsp/op/wrapper/op50_uniform_4ins.pkl"),
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
                pathlib.Path("test_dataset/routing/tsp/pctsp/task/pctsp50_uniform_task.pkl"),
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
                pathlib.Path("test_dataset/routing/tsp/pctsp/wrapper/pctsp50_uniform_4ins.pkl"),
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
                pathlib.Path("test_dataset/routing/tsp/spctsp/task/spctsp50_uniform_task.pkl"),
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
                pathlib.Path("test_dataset/routing/tsp/spctsp/wrapper/spctsp50_uniform_4ins.pkl"),
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
                pathlib.Path("test_dataset/routing/tsp/tsp/task/tsp50_cluster_task.pkl"),
                pathlib.Path("test_dataset/routing/tsp/tsp/task/tsp50_gaussian_task.pkl"),
            ]
            tsp_test_files_list_2 = [
                pathlib.Path("test_dataset/routing/tsp/tsp/task/tsp50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/tsp/tsp/task/tsp500_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in tsp_test_files_list_1:
                if test_file not in exclude_test_files:
                    task = TSPTask()
                    task.from_pickle(test_file)
                    task._normalize_data()
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
                pathlib.Path("test_dataset/routing/tsp/tsp/wrapper/tsp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/tsp/tsp/wrapper/tsp500_uniform_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in tsp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = TSPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrp_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrp/task/cvrp500_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrp/task/ovrp50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrp/task/ovrp100_uniform_task.pkl"),
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
                pathlib.Path("test_dataset/routing/vrp/cvrp/wrapper/cvrp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrp/wrapper/cvrp500_uniform_4ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrp/wrapper/ovrp50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrp/wrapper/ovrp100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrpb_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPBTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrpb_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/cvrpb50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/cvrpb100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/cvrpmb50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/cvrpmb100_uniform_task.pkl"),     
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/ovrpb50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/ovrpb100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/ovrpmb50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/task/ovrpmb100_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in cvrpb_test_files_list:
                if test_file not in exclude_test_files:
                    task = CVRPBTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrpb_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/cvrpb50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/cvrpb100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/cvrpmb50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/cvrpmb100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/ovrpb50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/ovrpb100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/ovrpmb50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpb/wrapper/ovrpmb100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrpb_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPBWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrpbl_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPBLTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrpbl_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/cvrpbl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/cvrpbl100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/cvrpmbl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/cvrpmbl100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/ovrpbl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/ovrpbl100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/ovrpmbl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/task/ovrpmbl100_uniform_task.pkl"), 
            ]
            task_list = list()
            for test_file in cvrpbl_test_files_list:
                if test_file not in exclude_test_files:
                    task = CVRPBLTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrpbl_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/cvrpbl50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/cvrpbl100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/cvrpmbl50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/cvrpmbl100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/ovrpbl50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/ovrpbl100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/ovrpmbl50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbl/wrapper/ovrpmbl100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrpbl_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPBLWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrpbltw_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPBLTWTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrpbltw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpbltw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpbltw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpmbltw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/cvrpmbltw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/ovrpbltw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/ovrpbltw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/ovrpmbltw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/task/ovrpmbltw100_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in cvrpbltw_test_files_list:
                if test_file not in exclude_test_files:
                    task = CVRPBLTWTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrpbltw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/cvrpbltw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/cvrpbltw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/cvrpmbltw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/cvrpmbltw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/ovrpbltw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/ovrpbltw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/ovrpmbltw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbltw/wrapper/ovrpmbltw100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrpbltw_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPBLTWWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrpbtw_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPBTWTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrpbtw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/cvrpbtw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/cvrpbtw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/cvrpmbtw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/cvrpmbtw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/ovrpbtw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/ovrpbtw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/ovrpmbtw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/task/ovrpmbtw100_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in cvrpbtw_test_files_list:
                if test_file not in exclude_test_files:
                    task = CVRPBTWTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrpbtw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/cvrpbtw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/cvrpbtw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/cvrpmbtw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/cvrpmbtw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/ovrpbtw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/ovrpbtw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/ovrpmbtw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpbtw/wrapper/ovrpmbtw100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrpbtw_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPBTWWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrpl_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPLTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrpl_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/cvrpl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/cvrpl100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/ovrpl50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/task/ovrpl100_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in cvrpl_test_files_list:
                if test_file not in exclude_test_files:
                    task = CVRPLTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrpl_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpl/wrapper/cvrpl50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/wrapper/cvrpl100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/wrapper/ovrpl50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpl/wrapper/ovrpl100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrpl_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPLWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrpltw_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPLTWTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrpltw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/task/cvrpltw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/task/cvrpltw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/task/ovrpltw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/task/ovrpltw100_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in cvrpltw_test_files_list:
                if test_file not in exclude_test_files:
                    task = CVRPLTWTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrpltw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/wrapper/cvrpltw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/wrapper/cvrpltw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/wrapper/ovrpltw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrpltw/wrapper/ovrpltw100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrpltw_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPLTWWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_cvrptw_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[CVRPTWTask]:
        # ``Solve`` mode
        if mode == "solve":
            cvrptw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrptw/task/cvrptw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrptw/task/cvrptw100_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrptw/task/ovrptw50_uniform_task.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrptw/task/ovrptw100_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in cvrptw_test_files_list:
                if test_file not in exclude_test_files:
                    task = CVRPTWTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list
        
        # ``Batch Solve`` mode
        if mode == "batch_solve":
            cvrptw_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/cvrptw/wrapper/cvrptw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrptw/wrapper/cvrptw100_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrptw/wrapper/ovrptw50_uniform_16ins.pkl"),
                pathlib.Path("test_dataset/routing/vrp/cvrptw/wrapper/ovrptw100_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in cvrptw_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = CVRPTWWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_mtvrp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[MTVRPTask]:
        # ``Solve`` mode
        if mode == "solve":
            mtvrp_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/mtvrp/task/mtvrp50_uniform_task.pkl"),
            ]
            task_list = list()
            for test_file in mtvrp_test_files_list:
                if test_file not in exclude_test_files:
                    task = MTVRPTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            mtvrp_test_files_list = [
                pathlib.Path("test_dataset/routing/vrp/mtvrp/wrapper/mtvrp50_uniform_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in mtvrp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MTVRPWrapper()
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
                pathlib.Path("test_dataset/graph/mcl/task/mcl_rb-large_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mcl/task/mcl_rb-small_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mcl/task/mcl_rb-small_uniform-weighted_task.pkl")
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
                pathlib.Path("test_dataset/graph/mcl/wrapper/mcl_rb-large_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mcl/wrapper/mcl_rb-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mcl/wrapper/mcl_rb-small_uniform-weighted_4ins.pkl"),
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
                pathlib.Path("test_dataset/graph/mcut/task/mcut_ba-large_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mcut/task/mcut_ba-small_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mcut/task/mcut_ba-small_uniform-weighted_task.pkl")
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
                pathlib.Path("test_dataset/graph/mcut/wrapper/mcut_ba-large_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mcut/wrapper/mcut_ba-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mcut/wrapper/mcut_ba-small_uniform-weighted_4ins.pkl"),
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
                pathlib.Path("test_dataset/graph/mis/task/mis_er-700-800_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mis/task/mis_rb-small_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mis/task/mis_rb-small_uniform-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mis/task/mis_satlib_no-weighted_task.pkl")
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
                pathlib.Path("test_dataset/graph/mis/wrapper/mis_er-700-800_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mis/wrapper/mis_rb-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mis/wrapper/mis_rb-small_uniform-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mis/wrapper/mis_satlib_no-weighted_4ins.pkl"),
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
                pathlib.Path("test_dataset/graph/mvc/task/mvc_rb-large_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mvc/task/mvc_rb-small_no-weighted_task.pkl"),
                pathlib.Path("test_dataset/graph/mvc/task/mvc_rb-small_uniform-weighted_task.pkl"),
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
                pathlib.Path("test_dataset/graph/mvc/wrapper/mvc_rb-large_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mvc/wrapper/mvc_rb-small_no-weighted_4ins.pkl"),
                pathlib.Path("test_dataset/graph/mvc/wrapper/mvc_rb-small_uniform-weighted_4ins.pkl"),
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
                pathlib.Path("test_dataset/portfolio/minvarpo/task/minvarpo_gbm_task.pkl"),
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
                pathlib.Path("test_dataset/portfolio/minvarpo/wrapper/minvarpo_gbm_4ins.pkl"),
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
                pathlib.Path("test_dataset/portfolio/maxretpo/task/maxretpo_gbm_task.pkl"),
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
                pathlib.Path("test_dataset/portfolio/maxretpo/wrapper/maxretpo_gbm_4ins.pkl"),
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
                pathlib.Path("test_dataset/portfolio/mopo/task/mopo_gbm_task.pkl"),
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
                pathlib.Path("test_dataset/portfolio/mopo/wrapper/mopo_gbm_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in mopo_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = MOPOWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    ########################################
    #            SAT Problems              #
    ########################################

    def _get_sata_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[SATATask]:
        # ``Solve`` mode
        if mode == "solve":
            sata_test_files_list = [
                pathlib.Path("test_dataset/sat/sata/task/sata_ca-small_task.pkl"),
            ]
            task_list = list()
            for test_file in sata_test_files_list:
                if test_file not in exclude_test_files:
                    task = SATATask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            sata_test_files_list = [
                pathlib.Path("test_dataset/sat/sata/wrapper/sata_ca-small_16ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in sata_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = SATAWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list

    def _get_satp_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[SATPTask]:
        # ``Solve`` mode
        if mode == "solve":
            satp_test_files_list = [
                pathlib.Path("test_dataset/sat/satp/task/satp_ca-small_task.pkl"),
            ]
            task_list = list()
            for test_file in satp_test_files_list:
                if test_file not in exclude_test_files:
                    task = SATPTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            satp_test_files_list = [
                pathlib.Path("test_dataset/sat/satp/wrapper/satp_ca-small_16ins.pkl")
            ]
            bacth_task_list = list()
            for test_file in satp_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = SATPWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
                    
    ########################################
    #            QAP Problems              #
    ########################################
    
    def _get_gm_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[GMTask]:
        # ``Solve`` mode
        if mode == "solve":
            gm_test_files_list = [
                pathlib.Path("test_dataset/qap/gm/task/gm_er_iso_task.pkl"),
                pathlib.Path("test_dataset/qap/gm/task/gm_er_sub_task.pkl"),
            ]
            task_list = list()
            for test_file in gm_test_files_list:
                if test_file not in exclude_test_files:
                    task = GMTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            gm_test_files_list = [
                pathlib.Path("test_dataset/qap/gm/wrapper/gm_er_iso_4ins.pkl"),
                pathlib.Path("test_dataset/qap/gm/wrapper/gm_er_sub_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in gm_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = GMWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list
        
    def _get_ged_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[GEDTask]:
        # ``Solve`` mode
        if mode == "solve":
            ged_test_files_list = [
                pathlib.Path("test_dataset/qap/ged/task/ged8_ncomp_task.pkl"),
                pathlib.Path("test_dataset/qap/ged/task/ged8_comp_task.pkl")
            ]
            task_list = list()
            for test_file in ged_test_files_list:
                if test_file not in exclude_test_files:
                    task = GEDTask()
                    task.from_pickle(test_file)
                    task_list.append(task)
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            ged_test_files_list = [
                pathlib.Path("test_dataset/qap/ged/wrapper/ged8_ncomp_4ins.pkl"),
            ]
            bacth_task_list = list()
            for test_file in ged_test_files_list:
                if test_file not in exclude_test_files:
                    wrapper = GEDWrapper()
                    wrapper.from_pickle(test_file)
                    bacth_task_list.append(wrapper.task_list)
            return bacth_task_list  

    ########################################
    #            EDA Problems              #
    ########################################

    def _get_edap_tasks(
        self, mode: str, exclude_test_files: List[pathlib.Path]
    ) -> List[EDAPTask]:
        # ``Solve`` mode
        if mode == "solve":
            from ml4co_kit import EDAP_ISPD2005Dataset
            dataset = EDAP_ISPD2005Dataset()
            task_list = [dataset.load(0)]
            return task_list

        # ``Batch Solve`` mode
        if mode == "batch_solve":
            from ml4co_kit import EDAP_ISPD2005Dataset
            dataset = EDAP_ISPD2005Dataset()
            task_list = [dataset.load(idx) for idx in range(8)]
            batch_task_list = [task_list]
            return batch_task_list