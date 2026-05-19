r"""
Base class for dataset testers.
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


from typing import Type
from ml4co_kit import DatasetBase


class DatasetTesterBase(object):
    def __init__(
        self, test_dataset_class: Type[DatasetBase],
    ):
        self.test_dataset_class = test_dataset_class
    
    def test(self):
        # Basic test
        self._test_load()

        # Other tests
        self._test_others()

    def _test_load(self):
        dataset = self.test_dataset_class()
        dataset.load(0)
                    
    def _test_others(self):
        raise NotImplementedError(
            "Subclasses should implement this method."
        )