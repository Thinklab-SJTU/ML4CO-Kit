r"""
Base class for generator testers.
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


from tqdm import tqdm
from typing import Type, List
from ml4co_kit import GeneratorBase


class GenTesterBase(object):
    def __init__(
        self, 
        test_gen_class: Type[GeneratorBase],
        test_args_list: List[dict]
    ):
        self.test_gen_class = test_gen_class
        self.test_args_list = test_args_list

    def test(self):
        # Test for each distribution type
        for test_args in tqdm(
            self.test_args_list, 
            desc=f"Testing {self.test_gen_class.__name__}"
        ):
            try:
                generator = self.test_gen_class(**test_args)
                generator.generate()
            except:
                raise ValueError(
                    f"Error occurred when testing {self.test_gen_class}\n"
                    f"Test args: {test_args} "
                )