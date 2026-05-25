r"""
Tester for CVRPB generator.
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


from ml4co_kit import CVRPBGenerator, CVRP_TYPE
from tests.generator_test.base import GenTesterBase


class CVRPBGenTester(GenTesterBase):
    def __init__(self):
        super(CVRPBGenTester, self).__init__(
            test_gen_class=CVRPBGenerator,
            test_args_list=[
                # Uniform
                {
                    "distribution_type": CVRP_TYPE.UNIFORM,
                    "cvrp_open": False,
                },
                # Uniform
                {
                    "distribution_type": CVRP_TYPE.UNIFORM,
                    "cvrp_open": True,
                },
                # Gaussian
                {
                    "distribution_type": CVRP_TYPE.GAUSSIAN,
                    "cvrp_open": False,
                },
                # Gaussian
                {
                    "distribution_type": CVRP_TYPE.GAUSSIAN,
                    "cvrp_open": True,
                },
            ]
        )