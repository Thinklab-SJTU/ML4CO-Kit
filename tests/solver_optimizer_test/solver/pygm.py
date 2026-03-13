r"""
Pygm Solver Tester.
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
from ml4co_kit import TASK_TYPE, PyGMSolver
from tests.solver_optimizer_test.base import SolverTesterBase
from ml4co_kit.extension.pygmtools import PyGMToolsQAPSolver


pygm_qap_solver_rrwm = PyGMToolsQAPSolver(solver_name="rrwm")
pygm_qap_solver_sm = PyGMToolsQAPSolver(solver_name="sm")
pygm_qap_solver_ipfp = PyGMToolsQAPSolver(solver_name="ipfp")
pygm_qap_solver_astar = PyGMToolsQAPSolver(solver_name="astar")


class PyGMSolverTester(SolverTesterBase):
    def __init__(self):
        super(PyGMSolverTester, self).__init__(
            mode_list=["solve"],
            test_solver_class=PyGMSolver,
            test_task_type_list=[
                TASK_TYPE.GM, TASK_TYPE.GM,  TASK_TYPE.GM, TASK_TYPE.GM,
                TASK_TYPE.GED,  TASK_TYPE.GED, TASK_TYPE.GED,  TASK_TYPE.GED
            ],
            test_args_list=[
                # GM
                {
                    "pygm_qap_solver":  pygm_qap_solver_astar  
                },
                {
                    "pygm_qap_solver":  pygm_qap_solver_rrwm
                } ,
                {
                    "pygm_qap_solver":  pygm_qap_solver_sm
                } ,
                {
                    "pygm_qap_solver":  pygm_qap_solver_ipfp
                } ,
                # GED
                {
                    "pygm_qap_solver":  pygm_qap_solver_astar
                } ,
                {
                    "pygm_qap_solver":  pygm_qap_solver_rrwm
                } ,
                {
                    "pygm_qap_solver":  pygm_qap_solver_sm
                } ,
                {
                    "pygm_qap_solver":  pygm_qap_solver_ipfp
                } ,
            ],
            exclude_test_files_list=[
                # GM
                {
                    pathlib.Path("test_dataset/qap/gm/task/gm_er_iso_task.pkl"),
                    pathlib.Path("test_dataset/qap/gm/task/gm_er_sub_task.pkl"),
                },
                {},
                {}, 
                {},
                # GED
                {
                    pathlib.Path("test_dataset/qap/ged/task/ged8_ncomp_task.pkl"),
                },
                {
                    pathlib.Path("test_dataset/qap/ged/task/ged8_comp_task.pkl"),
                },
                {
                    pathlib.Path("test_dataset/qap/ged/task/ged8_comp_task.pkl"),
                },
                {
                    pathlib.Path("test_dataset/qap/ged/task/ged8_comp_task.pkl"),
                }
            ]
        )
        
    def pre_test(self):
        pass