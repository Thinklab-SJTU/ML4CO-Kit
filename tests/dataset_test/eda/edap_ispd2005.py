r"""
Test EDAP_ISPD2005 Dataset.
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


class EDAP_ISPD2005DatasetTester(DatasetTesterBase):
    def __init__(self):
        super(EDAP_ISPD2005DatasetTester, self).__init__(
            test_dataset_class=EDAP_ISPD2005Dataset
        )

    def _test_others(self):
        dataset = EDAP_ISPD2005Dataset()
        edap_task = dataset.load(0)
        result = edap_task.evaluate(edap_task.ref_sol)
        print(f"EDAP_ISPD2005 {edap_task.name}: {result}")