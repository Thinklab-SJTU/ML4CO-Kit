r"""
MCL Wrapper.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import pathlib
import numpy as np
from typing import Union, List
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.graph.mcl import MClTask
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path


class MClWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(MClWrapper, self).__init__(
            task_type=TASK_TYPE.MCL, precision=precision
        )
        self.task_list: List[MClTask] = list()
        
    def from_txt(
        self, 
        file_path: pathlib.Path,
        ref: bool = False,
        overwrite: bool = True,
        show_time: bool = False
    ):
        """Read task data from ``.txt`` file"""
        # Overwrite task list if overwrite is True
        if overwrite:
            self.task_list: List[MClTask] = list()
        
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                
                # Case1: Node-weighted
                if "weights" in line:
                    # Parse
                    split_line = line.split(" weights ")
                    edge_index = split_line[0]
                    split_line = split_line[1].split(" label ")                    
                    nodes_weight = split_line[0]
                    sol = split_line[1]
                    
                    edge_index = edge_index.split(" ")
                    edge_index = np.array(
                        [
                            [int(edge_index[i]), int(edge_index[i + 1])]
                            for i in range(0, len(edge_index), 2)
                        ]
                    ).T
                    
                    nodes_weight = nodes_weight.split(" ")
                    nodes_weight = np.array(
                        [float(node_weight) for node_weight in nodes_weight],
                        dtype=self.precision
                    )
                    
                    sol = sol.split(" ")
                    sol = np.array([int(nodel_label) for nodel_label in sol])
                    
                    # Use ``from_data``
                    if overwrite:
                        mcl_task = MClTask(
                            node_weighted=True, precision=self.precision
                        )
                    else:
                        mcl_task = self.task_list[idx]
                    mcl_task.from_data(
                        edge_index=edge_index, nodes_weight=nodes_weight,
                        sol=sol, ref=ref
                    )
                    
                # Case2: Unweighted
                else:
                    # Parse
                    split_line = line.split(" label ")
                    edge_index = split_line[0]
                    sol = split_line[1]
                    
                    edge_index = edge_index.split(" ")
                    edge_index = np.array(
                        [
                            [int(edge_index[i]), int(edge_index[i + 1])]
                            for i in range(0, len(edge_index), 2)
                        ]
                    ).T
                    
                    sol = sol.split(" ")
                    sol = np.array([int(nodel_label) for nodel_label in sol])
                    
                    # Use ``from_data``
                    if overwrite:
                        mcl_task = MClTask(
                            node_weighted=False, precision=self.precision
                        )
                    else:
                        mcl_task = self.task_list[idx]
                    mcl_task.from_data(
                        edge_index=edge_index, sol=sol, ref=ref
                    )
                
                # Add to task list
                if overwrite:
                    self.task_list.append(mcl_task)
    
    def to_txt(
        self, file_path: pathlib.Path, show_time: bool = False, mode: str = "w"
    ):
        """Write task data to ``.txt`` file"""
        # Check file path
        check_file_path(file_path)
        
        # Save task data to ``.txt`` file
        with open(file_path, mode) as f:
            write_msg = f"Writing data to {file_path}"
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                # Check data and get variables
                task._check_edges_index_not_none()
                task._check_sol_not_none()
                edge_index = task.edge_index.T
                sol = task.sol
                
                # Write data to ``.txt`` file
                f.write(" ".join(str(src) + str(" ") + str(tgt) for src, tgt in edge_index))
                if task.node_weighted:
                    f.write(str(" ") + str("weights") + str(" "))
                    f.write(str(" ").join(str(node_weight) for node_weight in task.nodes_weight))
                f.write(str(" ") + str("label") + str(" "))
                f.write(str(" ").join(str(node_label) for node_label in sol))
                f.write("\n")
            f.close()
    
    def from_gpickle_result_folder(
        self, 
        graph_folder_path: pathlib.Path = None,
        result_foler_path: pathlib.Path = None,
        ref: bool = False,
        overwrite: bool = True,
        show_time: bool = False                  
    ):
        """Read task data from folder (to support NetworkX format)"""
        # Overwrite task list if overwrite is True
        if overwrite:
            self.task_list: List[MClTask] = list()
        
        # Check inconsistent file names between graph and result files
        if graph_folder_path is not None and result_foler_path is not None:
            graph_files = os.listdir(graph_folder_path)
            graph_files.sort()
            result_files = os.listdir(result_foler_path)
            result_files.sort()
            graph_name_list = [file.split(".")[0] for file in graph_files]
            result_name_list = [file.split(".")[0] for file in result_files]
            if graph_name_list != result_name_list:
                raise ValueError("Inconsistent file names between graph and result files.")
            
        # Get file paths and number of instances
        num_instance = None
        if graph_folder_path is not None:
            graph_files = os.listdir(graph_folder_path)
            graph_files.sort()
            graph_files_path = [
                os.path.join(graph_folder_path, file) 
                for file in graph_files if file.endswith((".gpickle"))
            ]
            num_instance = len(graph_files_path)
        if result_foler_path is not None:
            result_files = os.listdir(result_foler_path)
            result_files.sort()
            result_files_path = [
                os.path.join(result_foler_path, file) 
                for file in result_files if file.endswith((".result"))
            ]
            num_instance = len(result_files_path)
        
        # Set None to file paths if not provided
        if num_instance is None:
            raise ValueError(
                "``graph_folder_path`` and ``result_foler_path`` cannot be None at the same time."
            )
        elif num_instance == 0:
            raise ValueError("No instance found in the folder.")
        else:
            if graph_folder_path is None:
                graph_files_path = [None] * num_instance
            if result_foler_path is None:
                result_files_path = [None] * num_instance
        
        # Read task data from files
        if graph_folder_path is None:
            load_msg = f"Loading result from {result_foler_path}"
        else:
            if result_foler_path is None:
                load_msg = f"Loading data from {graph_folder_path}"
            else:
                load_msg = (
                    f"Loading data from {graph_folder_path} and "
                    f"result from {result_foler_path}"
                )
        
        for idx, (graph_file_path, result_file_path) in tqdm_by_time(
            enumerate(zip(graph_files_path, result_files_path)), load_msg, show_time
        ):
            if overwrite:
                mcl_task = MClTask(node_weighted=None, precision=self.precision)
            else:
                mcl_task = self.task_list[idx]
            mcl_task.from_gpickle_result(
                gpickle_file_path=graph_file_path, 
                result_file_path=result_file_path, 
                ref=ref
            )
            if overwrite:
                self.task_list.append(mcl_task)
        
    def to_gpickle_result_folder(
        self, 
        graph_folder_path: pathlib.Path = None, 
        result_foler_path: pathlib.Path = None, 
        show_time: bool = False,
        sequential_orderd: bool = True
    ):
        """Write task data to NetworkX format files"""
        # Write problem of task data (.gpickle)
        if graph_folder_path is not None:
            os.makedirs(graph_folder_path, exist_ok=True)
            write_msg = f"Writing data to {graph_folder_path}"
            idx = 1  # Initialize idx
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                if sequential_orderd:
                    idx_str = f"{idx:08d}"
                    graph_file_path = os.path.join(graph_folder_path, f"{idx_str}.gpickle")
                    idx += 1  # Increment idx for the next task
                else:
                    graph_file_path = os.path.join(graph_folder_path, f"{task.name}.gpickle")
                task.to_gpickle_result(gpickle_file_path=graph_file_path)
        
        # Write result of task data (.result)
        if result_foler_path is not None:
            os.makedirs(result_foler_path, exist_ok=True)
            write_msg = f"Writing result to {result_foler_path}"
            idx = 1  # Initialize idx
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                if sequential_orderd:
                    idx_str = f"{idx:08d}"
                    result_file_path = os.path.join(result_foler_path, f"{idx_str}.result")
                    idx += 1  # Increment idx for the next task
                else:
                    result_file_path = os.path.join(result_foler_path, f"{task.name}.result")
                task.to_gpickle_result(result_file_path=result_file_path)