r"""
Test Dataset Module.
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
import sys
from typing import Type
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)


# Dataset Testers
from tests.dataset_test import (
    DatasetTesterBase,
    EDAP_ISPD2005DatasetTester,
    EDAP_ISPD2005FreeDatasetTester,
    EDAP_MMSDatasetTester,
    GED_AIDS700nefDatasetTester
)
dataset_tester_list = [
    EDAP_ISPD2005DatasetTester,
    EDAP_ISPD2005FreeDatasetTester,
    EDAP_MMSDatasetTester,
    GED_AIDS700nefDatasetTester
]


# Test Dataset
def test_dataset():
    for test_class in dataset_tester_list:
        test_class: Type[DatasetTesterBase]
        test_class().test()
    

# Main
if __name__ == "__main__":
    test_dataset()