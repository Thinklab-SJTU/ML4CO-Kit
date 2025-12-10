r"""
Tester for MVC generator.
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


from ml4co_kit import MVCGenerator, GRAPH_TYPE
from ml4co_kit.generator.graph.base import (
    GraphWeightGenerator, GRAPH_WEIGHT_TYPE
)
from tests.generator_test.base import GenTesterBase


class MVCGenTester(GenTesterBase):
    def __init__(self):
        super(MVCGenTester, self).__init__(
            test_gen_class=MVCGenerator,
            test_args_list=[
                # Uniform (w uniform weighted)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "node_weighted": True,
                    "node_weighted_gen": GraphWeightGenerator(
                        weighted_type=GRAPH_WEIGHT_TYPE.UNIFORM),
                },
                # Uniform (w gaussian weighted)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "node_weighted": True,
                    "node_weighted_gen": GraphWeightGenerator(
                        weighted_type=GRAPH_WEIGHT_TYPE.GAUSSIAN),
                },
                # Uniform (w poisson weighted)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "node_weighted": True,
                    "node_weighted_gen": GraphWeightGenerator(
                        weighted_type=GRAPH_WEIGHT_TYPE.POISSON),
                },
                # Uniform (w exponential weighted)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "node_weighted": True,
                    "node_weighted_gen": GraphWeightGenerator(
                        weighted_type=GRAPH_WEIGHT_TYPE.EXPONENTIAL),
                },
                # Uniform (w lognormal weighted)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "node_weighted": True,
                    "node_weighted_gen": GraphWeightGenerator(
                        weighted_type=GRAPH_WEIGHT_TYPE.LOGNORMAL),
                },
                # Uniform (w powerlaw weighted)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "node_weighted": True,
                    "node_weighted_gen": GraphWeightGenerator(
                        weighted_type=GRAPH_WEIGHT_TYPE.POWERLAW),
                },
                # Uniform (w binomial weighted)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "node_weighted": True,
                    "node_weighted_gen": GraphWeightGenerator(
                        weighted_type=GRAPH_WEIGHT_TYPE.BINORMIAL),
                },
                # Watts-Strogatz (w/o weighted)
                {
                    "distribution_type": GRAPH_TYPE.WS,
                    "node_weighted": False,
                },
                # Barabasi-Albert (w/o weighted)
                {
                    "distribution_type": GRAPH_TYPE.BA,
                    "node_weighted": False,
                },
                # Holme-Kim (w/o weighted)
                {
                    "distribution_type": GRAPH_TYPE.HK,
                    "node_weighted": False,
                },
                # RB (w/o weighted)
                {
                    "distribution_type": GRAPH_TYPE.RB,
                    "node_weighted": False,
                },
            ]
        )