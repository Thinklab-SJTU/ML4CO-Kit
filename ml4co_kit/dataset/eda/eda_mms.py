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
from ml4co_kit.dataset.base import DatasetBase


class MMSDataset(DatasetBase):
    """
    @inproceedings{
        yan2009handling,
        title={Handling complexities in modern large-scale mixed-size placement},
        author={Yan, Jackey Z and Viswanathan, Natarajan and Chu, Chris},
        booktitle={Proceedings of the 46th Annual Design Automation Conference},
        pages={436--441},
        year={2009}
    }
    """
    
    def __init__(
        self, 
        extend_die: bool = True,
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        # Super Initialization  
        super(MMSDataset, self).__init__(
            task_type=TASK_TYPE.EDAP,
            dataset_name="MMS",
            precision=precision
        )

        # Initialize Attributes
        self.extend_die = extend_die
        
    def _preprocess(self):
       # Name list
        self.name_list = [
            "adaptec1", "adaptec2", "adaptec3", "adaptec4",
            "bigblue1", "bigblue2", "bigblue3", "bigblue4",
            "newblue1", "newblue2", "newblue3", "newblue4", 
            "newblue5", "newblue6", "newblue7",
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
            # (newblue1)
            np.array([
                [609, 616, 609+11174, 11776]
            ]), 
            # (newblue2)
            np.array([
                [7751, 32, 7751+19858, 128],
                [255, 128, 255+27354, 248],
                [32, 248, 32+27800, 22928],
                [255, 22928, 255+27354, 23132],
            ]), 
            # (newblue3)
            np.array([
                [7839, 39, 7839+30757, 135],
                [348, 135, 348+38248, 375],
                [36, 375, 36+38872, 49887],
                [348, 49887, 348+38248, 50223],
            ]), 
            # (newblue4)
            np.array([
                [7728, 32, 7728+10239, 128],
                [255, 128, 255+17712, 272],
                [24, 272, 24+18174, 18080],
                [255, 18080, 255+17712, 18320],
            ]), 
            # (newblue5)
            np.array([
                [7807, 67, 7807+17370, 139],
                [335, 139, 335+24842, 355],
                [36, 355, 36+25440, 25339],
                [335, 25339, 335+24842, 25627],
            ]), 
            # (newblue6)
            np.array([
                [7807, 76, 7807+19620, 148],
                [335, 148, 335+27092, 364],
                [36, 364, 36+27690, 27580],
                [335, 27580, 335+27092, 27868],
            ]), 
            # (newblue7)
            np.array([
                [7807, 49, 7807+30870, 121],
                [335, 121, 335+38342, 337],
                [36, 337, 36+38940, 39145],
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
        task_data.from_mms(
            name=name, die=die, root_path=self.extracted_save_path
        )
        return task_data