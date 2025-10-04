r"""
Base class for wrapper testers.
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


import os
import uuid
import pathlib
from typing import Type, List
from ml4co_kit import TaskBase, get_md5


class TaskTesterBase(object):
    def __init__(
        self, 
        test_task_class: Type[TaskBase],
        pickle_files_list: List[pathlib.Path],
    ):
        self.test_task_class = test_task_class
        self.pickle_files_list = pickle_files_list
    
    def test(self):
        # Create tmp folder
        os.makedirs("tmp", exist_ok=True)
        
        # Test for pickle & txt
        self._test_pickle_evaluate()

        # Test for other read and write methods
        self._test_other_rw_methods()
        
        # Test for visualization
        self._test_render()
    
    def _make_tmp_file(self) -> str:
        uuid_str = uuid.uuid4().hex
        tmp_file_path = f"tmp/tmp_{uuid_str}"
        return tmp_file_path
        
    def _test_pickle_evaluate(self):
        for pkl_file in self.pickle_files_list:
            # Load task
            task = self.test_task_class()
            task.from_pickle(pkl_file)
            
            # Test for pickle
            tmp_pkl_file_path = self._make_tmp_file() + ".pkl"
            task.to_pickle(pathlib.Path(tmp_pkl_file_path))
            if get_md5(pkl_file) != get_md5(tmp_pkl_file_path):
                raise ValueError(f"Test for pickle {pkl_file} failed")
            os.remove(tmp_pkl_file_path)

            # Test for evaluate
            task.sol = task.ref_sol
            cost = task.evaluate()
            _ = task.evaluate_w_gap()
            print(f"{self.test_task_class.__name__}: {cost}")
            
    def _test_other_rw_methods(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _test_render(self):
        raise NotImplementedError("Subclasses should implement this method.")