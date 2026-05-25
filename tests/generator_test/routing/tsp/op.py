
r"""
Tester for OP generator.
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


from ml4co_kit import OPGenerator, OP_TYPE
from tests.generator_test.base import GenTesterBase


class OPGenTester(GenTesterBase):
    def __init__(self):
        super(OPGenTester, self).__init__(
            test_gen_class=OPGenerator,
            test_args_list=[
                # Uniform
                {
                    "distribution_type": OP_TYPE.UNIFORM,
                },
                # Constant
                {
                    "distribution_type": OP_TYPE.CONSTANT,
                },
                # Distance
                {
                    "distribution_type": OP_TYPE.DISTANCE,
                },
            ]
        )