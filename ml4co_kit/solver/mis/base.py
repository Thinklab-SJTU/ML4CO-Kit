r"""
Basic solver for Maximum Independent Set (MIS). 

MIS involves finding the largest set of vertices in an undirected 
graph such that no two vertices in the set are adjacent.
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


import os
import pickle
import numpy as np
import networkx as nx
from typing import List
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.utils.graph.mis import MISGraphData
from ml4co_kit.utils.type_utils import TASK_TYPE, SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, iterative_execution_for_file


class MISSolver(SolverBase):
    def __init__(
        self, 
        solver_type: SOLVER_TYPE = None, 
        weighted: bool = False, 
        time_limit: float = 60.0
    ):
        super(MISSolver, self).__init__(
            task_type=TASK_TYPE.MIS, solver_type=solver_type
        )
        self.weighted = weighted
        self.time_limit = time_limit
        self.graph_data: List[MISGraphData] = list()

    def _check_edge_index_not_none(self):
        message = (
            f"``edge_index`` cannot be None! You can load ``edge_index`` using the "
            "methods including ``from_graph_data``, ``from_adj_martix``, ``from_txt``, "
            "``from_gpickle_result`` or ``from_gpickle_result_folder`` to obtain them."
        )
        for graph in self.graph_data:
            graph: MISGraphData
            if graph.edge_index is None:
                raise ValueError(message)

    def _check_nodes_label_not_none(self, ref: bool):
        msg = "ref_nodes_label" if ref else "nodes_label"
        message = (
            f"``{msg}`` cannot be None! You can use solvers based on ``MISSolver`` "
            "like ``KaMISSolver`` or use methods including ``from_graph_data``, "
            "``from_adj_martix``, ``from_txt``, ``from_gpickle_result`` or "
            "``from_gpickle_result_folder`` to obtain them."
        )  
        if ref:
            for graph in self.graph_data:
                graph: MISGraphData
                if graph.ref_nodes_label is None:
                    raise ValueError(message)
        else:
            for graph in self.graph_data:
                graph: MISGraphData
                if graph.nodes_label is None:
                    raise ValueError(message)

    def set_ref_as_solution(self):
        for graph in self.graph_data:
            graph: MISGraphData
            graph.nodes_label = graph.ref_nodes_label
            graph.sel_nodes_num = graph.ref_sel_nodes_num

    def set_solution_as_ref(self):
        for graph in self.graph_data:
            graph: MISGraphData
            graph.ref_nodes_label = graph.nodes_label
            graph.ref_sel_nodes_num = graph.sel_nodes_num
    
    def from_gpickle_result(
        self, 
        gpickle_file_path: str = None,
        result_file_path: str = None,
        self_loop: bool = True,
        ref: bool = True,
        cover: bool = True
    ):
        # cover or not
        if cover:
            graph = MISGraphData()
        else:
            if len(self.graph_data) != 1:
                raise ValueError(
                    "Read data from only one graph, but save more than one piece of data"
                )
            graph: MISGraphData = self.graph_data[0]
        
        # read graph data
        if gpickle_file_path is not None:
            graph.from_gpickle(file_path=gpickle_file_path, self_loop=self_loop)
        if result_file_path is not None:
            graph.from_result(file_path=result_file_path, ref=ref)
        self.graph_data = [graph]
            
    def from_gpickle_result_folder(
        self, 
        gpickle_folder_path: str = None,
        result_folder_path: str = None,
        weighted: bool = False,
        self_loop: bool = True,
        ref: bool = False,
        cover: bool = True,
        show_time: bool = False
    ):
        # weighted
        if weighted == True:
            raise NotImplementedError(
                "The current version does not currently support weighted graphs"
            )
        
        # init
        gpickle_flag = False if gpickle_folder_path is None else True
        result_flag = False if result_folder_path is None else True
        
        # cover or not
        if cover:
            self.graph_data = list()
              
        # only data
        if gpickle_flag and not result_flag:
            files = os.listdir(gpickle_folder_path)
            files.sort()
            load_msg = f"Loading data from {gpickle_folder_path}"
            for idx, file_name in iterative_execution_for_file(
                enumerate(files), load_msg, show_time
            ):
                # file path and check format
                gpickle_file_path = os.path.join(gpickle_folder_path, file_name)
                if not gpickle_file_path.endswith(".gpickle"):
                    continue
                
                # cover or not
                graph = self.graph_data[idx] if not cover else MISGraphData()

                # read graph data
                graph.from_gpickle(file_path=gpickle_file_path, self_loop=self_loop)
                
                # cover or not
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
                
        # only solution
        if not gpickle_flag and result_flag:
            files = os.listdir(result_folder_path)
            files.sort()
            load_msg = f"Loading solutions from {result_folder_path}"
            for idx, file_name in iterative_execution_for_file(
                enumerate(files), load_msg, show_time
            ):
                # file path and check format
                result_file_path = os.path.join(result_folder_path, file_name)
                if not result_file_path.endswith(".result"):
                    continue
                
                # cover or not
                graph = self.graph_data[idx] if not cover else MISGraphData()

                # read graph data
                graph.from_result(file_path=result_file_path, ref=ref)

                # cover or not
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
            
        # both data and solutions [must have the same filename]
        if gpickle_flag and result_flag:
            files = os.listdir(gpickle_folder_path)
            files.sort()
            load_msg = f"Loading data from {gpickle_folder_path}"
            for idx, file_name in iterative_execution_for_file(
                enumerate(files), load_msg, show_time
            ):
                # file path and check format
                file_name: str
                gpickle_file_path = os.path.join(gpickle_folder_path, file_name)
                if not gpickle_file_path.endswith(".gpickle"):
                    continue
                result_file_path = os.path.join(
                    result_folder_path, 
                    file_name.replace(
                        ".gpickle", f"_{'weighted' if weighted else 'unweighted'}.result"
                    )
                )
                
                # cover or not
                graph = self.graph_data[idx] if not cover else MISGraphData()

                # read graph data
                graph.from_gpickle(file_path=gpickle_file_path, self_loop=self_loop)
                graph.from_result(file_path=result_file_path, ref=ref)
                
                # cover or not
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
            
    def from_txt(
        self,
        file_path: str, 
        ref: bool = False, 
        cover: bool = True, 
        show_time: bool = False
    ):
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # cover or not
        if cover:
            self.graph_data = list()
              
        # read the data form .txt
        with open(file_path, "r") as file:
            idx = 0
            
            # read by lines
            for line in iterative_execution_for_file(file, "Loading", show_time):
                # read data from one line
                line = line.strip()
                split_line = line.split(" label ")
                edge_index = split_line[0]
                nodes_label = split_line[1]
                edge_index = edge_index.split(" ")
                edge_index = np.array(
                    [
                        [int(edge_index[i]), int(edge_index[i + 1])]
                        for i in range(0, len(edge_index), 2)
                    ]
                ).T
                nodes_label = nodes_label.split(" ")
                nodes_label = np.array([int(nodel_label) for nodel_label in nodes_label])

                # cover or not
                graph = self.graph_data[idx] if not cover else MISGraphData()
                
                # load data
                graph.from_data(edge_index=edge_index, nodes_label=nodes_label, ref=ref)
                
                # cover or not
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
                
                # update index
                idx += 1
    
    def from_txt_only_sel_nodes_num(
        self,
        file_path: str, 
        ref: bool = False, 
    ):
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                split_line = line.split(": ")
                sel_nodes_num = int(split_line[1])
                if ref:
                    self.graph_data[idx].ref_sel_nodes_num = sel_nodes_num
                else:
                    self.graph_data[idx].sel_nodes_num = sel_nodes_num
                        
    def from_graph_data(
        self, 
        edge_index: List[np.ndarray] = None, 
        nodes_label: List[np.ndarray] = None, 
        ref: bool = False,
        cover: bool = True
    ):
        # init
        data_flag = False if edge_index is None else True
        result_flag = False if nodes_label is None else True

        # cover or not
        if cover:
            self.graph_data = list()

        # only data
        if data_flag and not result_flag:
            for idx in range(len(edge_index)):
                graph = self.graph_data[idx] if not cover else MISGraphData()
                graph.from_data(edge_index=edge_index[idx])
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
                    
        # only data
        if not data_flag and result_flag:
            for idx in range(len(nodes_label)):
                graph = self.graph_data[idx] if not cover else MISGraphData()
                graph.from_data(nodes_label=nodes_label[idx], ref=ref)
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
                    
        # both data and solutions
        if data_flag and result_flag:
            if not len(edge_index) == len(nodes_label):
                raise ValueError("The number of problems and solutions does not match!")
            
            for idx in range(len(nodes_label)):
                graph = self.graph_data[idx] if not cover else MISGraphData()
                graph.from_data(
                    edge_index=edge_index[idx], nodes_label=nodes_label[idx], ref=ref
                )
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
            
    def from_adj_matrix(
        self, 
        adj_matrix: List[np.ndarray],
        nodes_label: List[np.ndarray] = None, 
        self_loop: bool = True, 
        ref: bool = False,
        cover: bool = True
    ):
        # init
        result_flag = False if nodes_label is None else True

        # cover or not
        if cover:
            self.graph_data = list()

        # only data
        if not result_flag:
            for idx in range(len(adj_matrix)):
                graph = self.graph_data[idx] if not cover else MISGraphData()
                graph.from_adj_martix(adj_matrix[idx], self_loop=self_loop)
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
                    
        # both data and solutions
        else:
            if not len(adj_matrix) == len(nodes_label):
                raise ValueError("The number of problems and solutions does not match!")
            
            for idx in range(len(nodes_label)):
                graph = self.graph_data[idx] if not cover else MISGraphData()
                graph.from_data(nodes_label=nodes_label[idx], ref=ref)
                graph.from_adj_martix(adj_matrix[idx], self_loop=self_loop)
                if cover:
                    self.graph_data.append(graph)
                else:
                    self.graph_data[idx] = graph
    
    def from_nx_graph(self, nx_graphs: List[nx.Graph]):
        self.graph_data = list()
        for idx in range(len(nx_graphs)):
            graph = MISGraphData()
            graph.from_nx_graph(nx_graphs[idx])
            self.graph_data.append(graph)
        
    def to_gpickle_result_folder(
        self,
        gpickle_save_dir: str = None,
        gpickle_filename: str = None,
        result_save_dir: str = None,
        result_filename: str = None,
        weighted: bool = False,
        show_time: bool = False
    ):
        # weighted
        if weighted == True:
            raise NotImplementedError(
                "The current version does not currently support weighted graphs"
            )
        
        # .gpickle files
        if gpickle_save_dir is not None:
            # preparation
            if gpickle_filename.endswith(".gpickle"):
                gpickle_filename = gpickle_filename.replace(".gpickle", "")
            samples = len(self.graph_data)
            
            # makedirs
            if not os.path.exists(gpickle_save_dir):
                os.makedirs(gpickle_save_dir)

            write_msg = f"Writing gpickle files to {gpickle_save_dir}"
            for idx in iterative_execution(range, samples, write_msg, show_time):
                # file name & save path
                if samples == 1:
                    name = gpickle_filename + f".gpickle"
                else:
                    name = gpickle_filename + f"-{idx}.gpickle"
                save_path = os.path.join(gpickle_save_dir, name)
                
                # graph_data -> nx.Graph
                graph: MISGraphData = self.graph_data[idx]
                edge_index = graph.edge_index
                if not graph.self_loop:
                    self_loop: np.ndarray = np.arange(graph.nodes_num)
                    self_loop = self_loop.reshape(-1, 1).repeat(2, axis=1)
                    edge_index = np.concatenate([self_loop, edge_index.T], axis=0)
                    nx_graph: nx.Graph = nx.from_edgelist(edge_index)
                    nx_graph = nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
                else:
                    nx_graph = nx.from_edgelist(edge_index.T)

                # to pickle file
                with open(save_path, "wb") as f:
                    pickle.dump(nx_graph, f, pickle.HIGHEST_PROTOCOL)
            
        # .result files
        if result_save_dir is not None:
            # preparation
            if result_filename.endswith(".result"):
                result_filename = result_filename.replace(".result", "")
            samples = len(self.graph_data)
            
            # makedirs
            if not os.path.exists(result_save_dir):
                os.makedirs(result_save_dir)

            write_msg = f"Writing result files to {result_save_dir}"
            for idx in iterative_execution(range, samples, write_msg, show_time):
                # file name & save path
                if samples == 1:
                    if weighted:
                        name = result_filename + f"_weighted.result"
                    else:
                        name = result_filename + f"_unweighted.result"
                else:
                    if weighted:
                        name = result_filename + f"_weighted-{idx}.result"
                    else:
                        name = result_filename + f"_unweighted-{idx}.result"
                        
                save_path = os.path.join(result_save_dir, name)
                
                # write
                graph: MISGraphData = self.graph_data[idx]
                nodes_label = graph.nodes_label
                with open(save_path, "w") as f:
                    for node_label in nodes_label:
                        f.write(f"{node_label}\n")
                    f.close()
        
    def to_txt(self, file_path: str = "example.txt"):
        # check
        self._check_edge_index_not_none()
        self._check_nodes_label_not_none(ref=False)
        
        # write
        with open(file_path, "w") as f:
            for graph in self.graph_data:
                edge_index = graph.edge_index.T
                nodes_label = graph.nodes_label.astype(np.int32)
                f.write(" ".join(str(src) + str(" ") + str(tgt) for src, tgt in edge_index))
                f.write(str(" ") + str("label") + str(" "))
                f.write(str(" ").join(str(node_label) for node_label in nodes_label))
                f.write("\n")
            f.close()
    
    def evaluate(self, calculate_gap: bool = False, check_constraint: bool = True):
        if calculate_gap:
            snn_list = list()
            rsnn_list = list()
            gap_list = list()
            for graph in self.graph_data:
                snn, rsnn, gap = graph.evaluate(
                    calculate_gap=True, check_constraint=check_constraint
                )
                snn_list.append(snn)
                rsnn_list.append(rsnn)
                gap_list.append(gap)
            snn_avg = np.average(np.array(snn_list))    
            rsnn_avg = np.average(np.array(rsnn_list))
            gap = np.array(gap_list) 
            gap_avg = np.average(gap)
            gap_std = np.std(gap)   
            return snn_avg, rsnn_avg, gap_avg, gap_std
        else:
            snn_list = list()
            for graph in self.graph_data:
                snn = graph.evaluate(
                    calculate_gap=False, check_constraint=check_constraint
                )
                snn_list.append(snn)
            snn_avg = np.average(np.array(snn_list))   
            return snn_avg

    def solve(self):
        raise NotImplementedError(
            "The method ``solve`` is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "MISSolver"