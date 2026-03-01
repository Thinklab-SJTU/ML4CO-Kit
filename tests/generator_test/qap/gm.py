r"""
Tester for GM generator.
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


from ml4co_kit import GMGenerator, GM_TYPE, QAPGraphGenerator
from tests.generator_test.base import GenTesterBase


# QAP Graph Generator for different combinations of node/edge weights
qap_graph_gen_tt = QAPGraphGenerator(node_weighted=True, edge_weighted=True)
qap_graph_gen_tf = QAPGraphGenerator(node_weighted=True, edge_weighted=False)
qap_graph_gen_ft = QAPGraphGenerator(node_weighted=False, edge_weighted=True)
qap_graph_gen_ff = QAPGraphGenerator(node_weighted=False, edge_weighted=False)


# Tester for GM Generator with different QAP Graph Generators
class GMGenTester(GenTesterBase):
    def __init__(self):
        super(GMGenTester, self).__init__(
            test_gen_class=GMGenerator,
            test_args_list=[
                # Isomorphic (w node/edge weights)
                {
                    "distribution_type": GM_TYPE.ISO,
                    "qap_graph_generator": qap_graph_gen_tt,
                },
                # Isomorphic (w node weights)
                {
                    "distribution_type": GM_TYPE.ISO,
                    "qap_graph_generator": qap_graph_gen_tf,
                },
                # Isomorphic (w edge weights)
                {
                    "distribution_type": GM_TYPE.ISO,
                    "qap_graph_generator": qap_graph_gen_ft,
                },
                # Isomorphic (w no node/edge weights)
                {
                    "distribution_type": GM_TYPE.ISO,
                    "qap_graph_generator": qap_graph_gen_ff,
                },
                # Subgraph
                {
                    "distribution_type": GM_TYPE.SUB,
                    "qap_graph_generator": qap_graph_gen_tt,
                    "sub_ratio_scale": (0.3, 0.7),
                },
            ]
        )