r"""
Tester for MaxRetPO generator.
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


from ml4co_kit import MaxRetPOGenerator, PO_TYPE
from tests.generator_test.base import GenTesterBase


class MaxRetPOGenTester(GenTesterBase):
    def __init__(self):
        super(MaxRetPOGenTester, self).__init__(
            test_gen_class=MaxRetPOGenerator,
            test_args_list=[
                # GBM
                {
                    "distribution_type": PO_TYPE.GBM,
                },
                # Factor
                {
                    "distribution_type": PO_TYPE.FACTOR,
                },
                # VAR(1)
                {
                    "distribution_type": PO_TYPE.VAR1,
                },
                # MVT
                {
                    "distribution_type": PO_TYPE.MVT,
                },
                # GRACH
                {
                    "distribution_type": PO_TYPE.GRACH,
                },
                # Jump
                {
                    "distribution_type": PO_TYPE.JUMP,
                },
                # Regime
                {
                    "distribution_type": PO_TYPE.REGIME,
                },
            ]
        )