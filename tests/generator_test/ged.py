r"""
Tester for GED generator.
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

from ml4co_kit import GEDGenerator, GRAPH_TYPE
from ml4co_kit.generator.graphset.base import (
    GraphFeatureGenerator, GRAPH_FEATURE_TYPE
)
from ml4co_kit.generator.graphset.gm import GRAPH_GENERATE_RULE
from tests.generator_test.base import GenTesterBase

class GEDGenTester(GenTesterBase):
    def __init__(self):
        super(GEDGenTester, self).__init__(
            test_gen_class=GEDGenerator,
            test_args_list=[
                # Isomorphic graph problem
                # Uniform (node/edge uniform featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.ISOMORPHIC,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
                },
                # Uniform (node/edge gaussian featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.ISOMORPHIC,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.GAUSSIAN),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.GAUSSIAN),
                },
                # Uniform (node/edge poisson featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.ISOMORPHIC,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.POISSON),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.POISSON),
                },
                # Uniform (node/edge exponential featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.ISOMORPHIC,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.EXPONENTIAL),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.EXPONENTIAL),
                },
                # Uniform (node/edge lognormal featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.ISOMORPHIC,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.LOGNORMAL),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.LOGNORMAL),
                },
                # Uniform (node/edge powerlaw featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.ISOMORPHIC,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.POWERLAW),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.POWERLAW),
                },
                # Uniform (node/edge binomial featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.ISOMORPHIC,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.BINOMIAL),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.BINOMIAL),
                },
                
                # Induced subgraph problem
                # Uniform (node/edge uniform featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.INDUCED_SUBGRAPH,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
                },
                # Perturbed graph problem
                # Uniform (node/edge uniform featured)
                {
                    "distribution_type": GRAPH_TYPE.ER,
                    "graph_generate_rule": GRAPH_GENERATE_RULE.PERTURBED,
                    "node_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
                    "edge_feature_gen": GraphFeatureGenerator(
                        feature_type=GRAPH_FEATURE_TYPE.UNIFORM),
                },
            ]
        )