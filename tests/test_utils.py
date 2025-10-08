r"""
Test Task Module.
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
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)


# Checker
from ml4co_kit.utils.env_utils import EnvChecker
env_checker = EnvChecker()


# Utils Testers
from tests.utils_test import FileUtilsTester
test_class_list = [FileUtilsTester]


# if torch is supported
if env_checker.check_torch():
    from tests.utils_test import TypeUtilsTester
    test_class_list += [
        TypeUtilsTester
    ]


# Test Utils
def test_utils():
    for test_class in test_class_list:
        test_class().test()
    

# Main
if __name__ == "__main__":
    test_utils()