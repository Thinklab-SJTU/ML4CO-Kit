r"""
Tester for CVRP generator.
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


from ml4co_kit import CVRPGenerator, CVRP_TYPE
from tests.generator_test.base import GenTesterBase


class CVRPGenTester(GenTesterBase):
    def __init__(self):
        super(CVRPGenTester, self).__init__(
            test_gen_class=CVRPGenerator,
            test_args_list=[
                # Uniform
                {
                    "distribution_type": CVRP_TYPE.UNIFORM,
                },
                # Gaussian
                {
                    "distribution_type": CVRP_TYPE.GAUSSIAN,
                },
            ]
        )