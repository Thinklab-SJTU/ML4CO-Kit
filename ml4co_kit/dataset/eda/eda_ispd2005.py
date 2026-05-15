r"""
ISPD2005 Dataset for EDA.
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


import pathlib
import numpy as np
from typing import Union, List
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.eda.edap import EDAPTask
from ml4co_kit.task.eda.base import EDA_BENCH
from ml4co_kit.dataset.base import DatasetBase
from ml4co_kit.dataset.eda.c_ispd2005_reader import ISPD2005Reader


class ISPD2005Dataset(DatasetBase):
    """
    ISPD2005: https://www.ispd.cc/contests/05/contest.htm
    @inproceedings{
        nam2005ispd2005,
        title={The ISPD2005 placement contest and benchmark suite},
        author={Nam, Gi-Joon and Alpert, Charles J and Villarrubia, Paul and Winter, Bruce and Yildiz, Mehmet},
        booktitle={Proceedings of the 2005 international symposium on Physical design},
        pages={216--220},
        year={2005}
    }
    """
    
    def __init__(
        self, 
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        # Super Initialization  
        super(ISPD2005Dataset, self).__init__(
            task_type=TASK_TYPE.EDAP,
            dataset_name="ISPD2005",
            precision=precision
        )
        self.reader = ISPD2005Reader()

    def _preprocess(self):
       # Name list
        self.name_list = [
            "adaptec1", "adaptec2", "adaptec3", "adaptec4",
            "bigblue1", "bigblue2", "bigblue3", "bigblue4",
        ]

        # Die size
        self.die_list = [
            # (adaptec1)
            np.array([
                [459, 459, 459+10692, 11139]
            ]), 
            # (adaptec2)
            np.array([
                [609, 616, 609+14054, 14656]
            ]), 
            # (adaptec3)
            np.array([
                [7807, 58, 7807+15120, 130],
                [335, 130, 335+22592, 346],
                [36, 346, 36+23190, 23098],
                [335, 23098, 335+22592, 23386],
            ]), 
            # (adaptec4)
            np.array([
                [7807, 58, 7807+15120, 130],
                [335, 130, 335+22592, 346],
                [36, 346, 36+23190, 23098],
                [335, 23098, 335+22592, 23386],
            ]), 
            # (bigblue1)
            np.array([
                [459, 459, 459+10692, 11139]
            ]), 
            # (bigblue2)
            np.array([
                [7807, 76, 7807+10620, 148],
                [335, 148, 335+18092, 364],
                [36, 364, 36+18690, 18580],
                [335, 18580, 335+18092, 18868],
            ]), 
            # (bigblue3)
            np.array([
                [7807, 76, 7807+19620, 148],
                [335, 148, 335+27092, 364],
                [36, 364, 36+27690, 27580],
                [335, 27580, 335+27092, 27868],
            ]), 
            # (bigblue4)
            np.array([
                [7807, 58, 7807+24120, 130],
                [335, 130, 335+31592, 346],
                [36, 346, 36+32190, 32098],
                [335, 32098, 335+31592, 32386],
            ]), 
        ]

    def _from_nodes(self, file_path: pathlib.Path) -> List[np.ndarray]:
        return self.reader.from_nodes(str(file_path))

    def _from_nets(self, file_path: pathlib.Path) -> List[np.ndarray]:
        return self.reader.from_nets(str(file_path))

    def _from_lg_pl(self, file_path: pathlib.Path) -> np.ndarray:
        return self.reader.from_lg_pl(str(file_path))
        
    def _load(self, idx) -> EDAPTask:
        # Get basic information
        name = self.name_list[idx]
        die = self.die_list[idx]
        nodes_file_path = self.extracted_save_path / f"{name}/{name}.nodes"
        nets_file_path = self.extracted_save_path / f"{name}/{name}.nets"

        # Read data from files
        cells, macro_mask = self._from_nodes(nodes_file_path)
        cells_num = cells.shape[0]
        nets = self._from_nets(nets_file_path)

        # Create and return EDAPTask
        task_data = EDAPTask(
            precision=self.precision, benchmark_name=EDA_BENCH.ISPD2005
        )
        task_data.from_data(
            die=die, cells=cells, cells_num=cells_num, 
            macro_mask=macro_mask, nets=nets, name=name
        )

        # Add to cache
        aux_file_path = self.extracted_save_path / f"{name}/{name}.aux"
        task_data.cache["ispd2005_aux"] = aux_file_path
        return task_data