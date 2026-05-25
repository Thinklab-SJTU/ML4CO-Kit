r"""
Tester for ATSP generator.
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


from ml4co_kit import ATSPGenerator, ATSP_TYPE
from tests.generator_test.base import GenTesterBase


class ATSPGenTester(GenTesterBase):
    def __init__(self):
        super(ATSPGenTester, self).__init__(
            test_gen_class=ATSPGenerator,
            test_args_list=[
                # Uniform
                {
                    "distribution_type": ATSP_TYPE.UNIFORM,
                },
                # Gaussian
                {
                    "distribution_type": ATSP_TYPE.SAT,
                },
                # Cluster
                {
                    "distribution_type": ATSP_TYPE.HCP,
                },
            ]
        )