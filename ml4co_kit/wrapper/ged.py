r"""
GED Wrapper.
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
from ml4co_kit.task.graphset.ged import GEDTask
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path


r"""
GED Wrapper.
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
from ml4co_kit.task.graphset.base import Graph
from ml4co_kit.task.graphset.ged import GEDTask
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path


class GEDWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(GEDWrapper, self).__init__(
            task_type=TASK_TYPE.GED, precision=precision
        )
        self.task_list: List[GEDTask] = list()
        
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
            self.task_list: List[GEDTask] = list()
        
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                
                split_line = line.split("edge_index_for_graph1 ")
                tmp_line = split_line[1]
                split_line = tmp_line.split(" edge_index_for_graph2 ")
                edge_index1_str = split_line[0]
                tmp_line = split_line[1]
                
                split_line = tmp_line.split(" node_feature_for_graph1 ")
                edge_index2_str = split_line[0]   
                tmp_line = split_line[1]
                
                split_line = tmp_line.split(" node_feature_for_graph2 ")
                node_feat1_str = split_line[0]   
                tmp_line = split_line[1]
                
                split_line = tmp_line.split(" edge_feature_for_graph1 ")
                node_feat2_str = split_line[0]    
                tmp_line = split_line[1]

                split_line = tmp_line.split(" edge_feature_for_graph2 ")
                edge_feat1_str = split_line[0]    
                tmp_line = split_line[1]

                split_line = tmp_line.split(" edit_path ")
                edge_feat2_str = split_line[0] 
                sol_str = split_line[1]   
                
                edge_index1 = edge_index1_str.split(" ")
                edge_index2 = edge_index2_str.split(" ")
                edge_index1 = np.array(
                    [
                        [int(edge_index1[i]), int(edge_index1[i+1])]
                        for i in range(0, len(edge_index1), 2)
                    ]
                ).T
                edge_index2 = np.array(
                    [
                        [int(edge_index2[i]), int(edge_index2[i+1])]
                        for i in range(0, len(edge_index2), 2)
                    ]
                ).T
                
                node_feat1 = node_feat1_str.split(";")
                node_feat2 = node_feat2_str.split(";")
                node_feat1 = np.array(
                    [[float(v) for v in feat.split(" ")] for feat in node_feat1],
                    dtype=self.precision
                )
                node_feat2 = np.array(
                    [[float(v) for v in feat.split(" ")] for feat in node_feat2],
                    dtype=self.precision
                )
                
                edge_feat1 = edge_feat1_str.split(";")
                edge_feat2 = edge_feat2_str.split(";")
                edge_feat1 = np.array(
                    [[float(v) for v in feat.split(" ")] for feat in edge_feat1],
                    dtype = self.precision
                )
                edge_feat2 = np.array(
                    [[float(v) for v in feat.split(" ")] for feat in edge_feat2],
                    dtype = self.precision
                )
                
                sol = sol_str.split(" ")
                sol = np.array(
                    [float(edit_path) for edit_path in sol], 
                    dtype=self.precision
                ).reshape(node_feat1.shape[0]+1, node_feat2.shape[0]+1)
                
                graph1 = Graph(precision=self.precision)
                graph2 = Graph(precision=self.precision)
                graph1.from_data(edge_index=edge_index1, nodes_feature=node_feat1, edges_feature=edge_feat1)      
                graph2.from_data(edge_index=edge_index2, nodes_feature=node_feat2, edges_feature=edge_feat2)  
                graphs = [graph1, graph2]    
                
                if overwrite:
                    ged_task = GEDTask(precision=self.precision)
                else:
                    ged_task = self.task_list[idx]
                ged_task.from_data(graphs=graphs, sol=sol, ref=ref)
                
                # Add to task list
                if overwrite:
                    self.task_list.append(ged_task)
                
    
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
                task.graphs[0]._check_edges_index_not_none()
                task.graphs[1]._check_edges_index_not_none()
                task._check_sol_not_none()
                edge_index1 = task.graphs[0].edge_index.T
                edge_index2 = task.graphs[1].edge_index.T
                sol_rav = task.sol.ravel()
                
                # Write data to ``.txt`` file
                f.write(str(" ")+str("edge_index_for_graph1")+str(" "))
                f.write(" ".join(str(src) + str(" ") + str(tgt) for src, tgt in edge_index1))
                f.write(str(" ")+str("edge_index_for_graph2")+str(" "))
                f.write(" ".join(str(src) + str(" ") + str(tgt) for src, tgt in edge_index2))
            
                f.write(str(" ") + str("node_feature_for_graph1") + str(" "))
                f.write(";".join(" ".join(str(v) for v in feat) for feat in task.graphs[0].nodes_feature))
                f.write(str(" ") + str("node_feature_for_graph2") + str(" "))
                f.write(";".join(" ".join(str(v) for v in feat) for feat in task.graphs[1].nodes_feature))
                
                f.write(str(" ") + str("edge_feature_for_graph1") + str(" "))
                f.write(";".join(" ".join(str(v) for v in feat) for feat in task.graphs[0].edges_feature))
                f.write(str(" ") + str("edge_feature_for_graph2") + str(" "))
                f.write(";".join(" ".join(str(v) for v in feat) for feat in task.graphs[1].edges_feature))
                f.write(str(" ") + str("edit_path") + str(" "))
                f.write(str(" ").join(str(edit_path) for edit_path in sol_rav))
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
            self.task_list: List[GEDTask] = list()
        
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
                ged_task = GEDTask(precision=self.precision)
            else:
                ged_task = self.task_list[idx]
            ged_task.from_gpickle_result(
                gpickle_file_path=graph_file_path, 
                result_file_path=result_file_path, 
                ref=ref
            )
            if overwrite:
                self.task_list.append(ged_task)
        
    def to_gpickle_result_folder(
        self, 
        graph_folder_path: pathlib.Path = None, 
        result_foler_path: pathlib.Path = None, 
        show_time: bool = False
    ):
        """Write task data to NetworkX format files"""
        # Write problem of task data (.gpickle)
        if graph_folder_path is not None:
            os.makedirs(graph_folder_path, exist_ok=True)
            write_msg = f"Writing data to {graph_folder_path}"
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                graph_file_path = os.path.join(graph_folder_path, f"{task.name}.gpickle")
                task.to_gpickle_result(gpickle_file_path=graph_file_path)
        
        # Write result of task data (.result)
        if result_foler_path is not None:
            os.makedirs(result_foler_path, exist_ok=True)
            write_msg = f"Writing result to {result_foler_path}"
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                result_file_path = os.path.join(result_foler_path, f"{task.name}.result")
                task.to_gpickle_result(result_file_path=result_file_path)