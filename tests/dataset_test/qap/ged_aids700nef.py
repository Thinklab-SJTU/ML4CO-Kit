r"""
Test GED_AIDS700nef Dataset.
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


from tests.dataset_test.base import DatasetTesterBase
from ml4co_kit import *


class GED_AIDS700nefDatasetTester(DatasetTesterBase):
    def __init__(self):
        super(GED_AIDS700nefDatasetTester, self).__init__(
            test_dataset_class=GED_AIDS700nefDataset
        )

    def _test_others(self):
        pass