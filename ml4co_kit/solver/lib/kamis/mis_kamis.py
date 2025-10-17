r"""
KaMIS Algorithm for MIS
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


import pathlib
import tempfile
import subprocess
import numpy as np
import networkx as nx
from ml4co_kit.task.graph.mis import MISTask


def mis_kamis(
    task_data: MISTask, 
    kamis_time_limit: float, 
    kamis_weighted_scale: float = 1e5
):
    # Step1: Prepare for the solving
    nx_graph = task_data.to_networkx()
    weighted = task_data.node_weighted
    base_path = pathlib.Path(__file__).parent
    input_path = tempfile.NamedTemporaryFile(mode="w", delete=False)
    output_path = tempfile.NamedTemporaryFile(mode="w", delete=False)
    
    # Step2: Prepare the input graph
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()
    wt = 0 if not weighted else 10
    res = f"{n} {m} {wt}\n"
    if weighted:
        adj_matrix = task_data.to_adj_matrix()
        for idx, row in enumerate(adj_matrix):
            line = [int(task_data.nodes_weight[idx] * kamis_weighted_scale)]
            line = line + (np.where(row == 1)[0] + 1).tolist()
            res += " ".join(map(str, line)) + "\n"
    else:
        for n, nbrsdict in nx_graph.adjacency():
            line = []
            for nbr, _ in sorted(nbrsdict.items()):
                line.append(nbr + 1)
            res += " ".join(map(str, line)) + "\n"
    with open(input_path.name, "w") as res_file:
        res_file.write(res)

    # Step3: Executable
    if weighted:
        executable = (
            base_path / "KaMIS" / "deploy" / "weighted_branch_reduce"
        )
    else:
        executable = base_path / "KaMIS" / "deploy" / "redumis"

    # Step4: Solve the problem
    arguments = [
        input_path.name,  # input
        "--output",
        output_path.name,  # output
        "--time_limit",
        str(kamis_time_limit),
    ]
    message = (
        "[linux] Please check KaMIS compilation. You can try ``self.install() " 
        "If you are sure that the ``KaMIS`` is correct, please confirm "
        "whether the Conda environment of the terminal is consistent "
        "with the Python environment. \n"
        "[macos] Current version of ML4CO-Kit is not supported for KaMIS on macOS." 
    ) 
    try:
        subprocess.run(
            [executable] + arguments, shell=False, capture_output=True, text=True
        )
    except TypeError:
        raise TypeError(message)
    except FileNotFoundError:
        raise FileNotFoundError(message)
    
    # Step5: Call ``from_gpickle_result`` to get the result
    task_data.from_gpickle_result(result_file_path=output_path.name)