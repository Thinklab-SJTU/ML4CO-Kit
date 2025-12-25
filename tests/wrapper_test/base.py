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
from ml4co_kit import WrapperBase, GeneratorBase, SolverBase, get_md5


class WrapperTesterBase(object):
    def __init__(
        self, 
        test_wrapper_class: Type[WrapperBase],
        generator: GeneratorBase,
        solver: SolverBase,
        pickle_files_list: List[pathlib.Path],
        txt_files_list: List[pathlib.Path],
    ):
        self.test_wrapper_class = test_wrapper_class
        self.generator = generator
        self.solver = solver
        self.pickle_files_list = pickle_files_list
        self.txt_files_list = txt_files_list
    
    def test(self):
        # Create tmp folder
        os.makedirs("tmp", exist_ok=True)
        
        # Test for pickle & txt
        self._test_pickle_txt()
        
        # Test for other read and write methods
        self._test_other_rw_methods()
        
        # Test for generate
        self._test_generate()
        
        # Test for solve
        self._test_solve_evaluate()
    
    def _make_tmp_file(self) -> str:
        uuid_str = uuid.uuid4().hex
        tmp_file_path = f"tmp/tmp_{uuid_str}"
        return tmp_file_path
        
    def _test_pickle_txt(self):
        # Test overwrite in ``from_txt``
        wrapper = self.test_wrapper_class()
        wrapper.from_txt(self.txt_files_list[0], ref=False)
        wrapper.from_txt(self.txt_files_list[0], ref=True, overwrite=False)
        eval_result = wrapper.evaluate_w_gap()
        print(f"Test for overwrite in ``from_txt``: {eval_result}")
        
        # Test ``from_pickle`` and ``to_txt``
        for pkl_file, txt_file in zip(self.pickle_files_list, self.txt_files_list):
            wrapper = self.test_wrapper_class()
            wrapper.from_pickle(pkl_file)
            tmp_txt_file_path = self._make_tmp_file() + ".txt"
            wrapper.swap_sol_and_ref_sol()
            wrapper.to_txt(pathlib.Path(tmp_txt_file_path))
            if get_md5(txt_file) != get_md5(tmp_txt_file_path):
                raise ValueError(f"Test for pickle {pkl_file} failed")
            os.remove(tmp_txt_file_path)
    
    def _test_other_rw_methods(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _count_lines(self, file_path: pathlib.Path):
        with open(file_path, "r") as file:
            return len(file.readlines())
    
    def _test_generate(self):
        wrapper = self.test_wrapper_class()
        for num_threads in [1, 2]:
            for write_per_iters in [1, 2]:
                tmp_txt_file_path = self._make_tmp_file() + ".txt"
                wrapper.generate_w_to_txt(
                    file_path=tmp_txt_file_path,
                    generator=self.generator, 
                    solver=self.solver,
                    num_samples=4,
                    num_threads=num_threads,
                    batch_size=1,
                    write_per_iters=write_per_iters,
                    show_time=True
                )
                lines = self._count_lines(tmp_txt_file_path)
                if lines != 4:
                    raise ValueError(f"Test for generate {tmp_txt_file_path} failed")
                os.remove(tmp_txt_file_path)

    def _test_solve_evaluate(self):
        wrapper = self.test_wrapper_class()
        for txt_file in self.txt_files_list:
            for num_threads in [1, 2]:
                for show_time in [False, True]:
                    try:
                        wrapper.from_txt(txt_file, ref=True)
                        wrapper.solve(
                            solver=self.solver,
                            num_threads=num_threads,
                            show_time=show_time
                        )
                        sol_costs = wrapper.evaluate()
                        eval_result = wrapper.evaluate_w_gap()
                        msg = (
                            f"{self.test_wrapper_class.__name__} test on {txt_file} "
                            f"using {self.solver.solver_type.value}: {sol_costs}, "
                            f"compare with ref: {eval_result}"
                        )
                        print(msg)
                    except:
                        raise ValueError(f"Test for solve {txt_file} failed")