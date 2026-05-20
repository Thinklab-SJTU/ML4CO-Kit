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


import numpy as np
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.eda.edap import EDAPTask
from ml4co_kit.task.eda.base import EDA_BENCH
from ml4co_kit.dataset.base import DatasetBase


class EDAP_ISPD2005FreeDataset(DatasetBase):
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
        extend_die: bool = True,
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        # Super Initialization  
        super(EDAP_ISPD2005FreeDataset, self).__init__(
            task_type=TASK_TYPE.EDAP,
            dataset_name="ISPD2005Free",
            precision=precision
        )

        # Initialize Attributes
        self.extend_die = extend_die

    def _preprocess(self):
       # Name list
        self.name_list = [
            "adaptec1_allfree", "adaptec2_allfree", 
            "adaptec3_allfree", "adaptec4_allfree",
            "bigblue1_allfree", "bigblue2_allfree", 
            "bigblue3_allfree", "bigblue4_allfree",
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
        
    def _load(self, idx) -> EDAPTask:
        # Get original die
        name = self.name_list[idx]
        die = self.die_list[idx]

        # Extend die if required
        if self.extend_die:
            die = np.array([[
                die[:, 0].min(),
                die[:, 1].min(),
                die[:, 2].max(),
                die[:, 3].max(),
            ]], dtype=self.precision)

        # Create task data
        task_data = EDAPTask(precision=self.precision)
        task_data.from_ispd2005_like(
            name=name, die=die, 
            root_path=self.extracted_save_path,
            benchmark_name=EDA_BENCH.ISPD2005FREE
        )
        return task_data
