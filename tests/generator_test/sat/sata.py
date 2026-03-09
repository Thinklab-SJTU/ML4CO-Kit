r"""
Tester for SAT-A generator.
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


from ml4co_kit import SATAGenerator, SAT_TYPE
from tests.generator_test.base import GenTesterBase


class SATAGenTester(GenTesterBase):
    def __init__(self):
        super(SATAGenTester, self).__init__(
            test_gen_class=SATAGenerator,
            test_args_list=[
                # 3-SAT
                {
                    "distribution_type": SAT_TYPE.PHASE,
                },
                # SR Model
                {
                    "distribution_type": SAT_TYPE.SR,
                },
                # Community Attachment
                {
                    "distribution_type": SAT_TYPE.CA,
                },
                # Popularity Similarity
                {
                    "distribution_type": SAT_TYPE.PS,
                },
                # K-Clique
                {
                    "distribution_type": SAT_TYPE.K_CLIQUE,
                },
                # K-Domset
                {
                    "distribution_type": SAT_TYPE.K_DOMSET,
                },
                # K-Vercov
                {
                    "distribution_type": SAT_TYPE.K_VERCOV,
                },
            ]
        )