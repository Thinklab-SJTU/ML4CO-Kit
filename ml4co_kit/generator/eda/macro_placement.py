r"""
Generator for Macro Placement instances.

References:
    [1] Mirhoseini et al., "A graph placement methodology for fast chip design",
        Nature, 2021.
    [2] Cheng & Yan, "The art of macro placement", ISPD 2005 benchmark suite.
    [3] ICCAD 2015 Contest: Incremental Timing-driven Placement.
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


import numpy as np
from enum import Enum
from typing import Union, List
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.eda.macro_placement import MacroPlacementTask
from ml4co_kit.generator.eda.base import EDAGeneratorBase


class MACRO_PLACEMENT_TYPE(str, Enum):
    """Distribution types for macro placement generation."""
    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    MIXED_SIZE = "mixed_size"


class MacroPlacementGenerator(EDAGeneratorBase):
    """Generator for Macro Placement instances."""
    
    def __init__(
        self, 
        distribution_type: MACRO_PLACEMENT_TYPE = MACRO_PLACEMENT_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        macros_num: int = 20,
        nets_num: int = 30,
        canvas_width: float = 100.0,
        canvas_height: float = 100.0,
        macro_width_range: tuple = (5.0, 20.0),
        macro_height_range: tuple = (5.0, 20.0),
        net_degree_range: tuple = (2, 5),
        # special args for clustered
        cluster_nums: int = 3,
        cluster_std: float = 0.15,
        # special args for mixed_size
        large_macro_ratio: float = 0.3,
        large_macro_scale: float = 3.0,
    ):
        # Super Initialization
        super(MacroPlacementGenerator, self).__init__(
            task_type=TASK_TYPE.MACRO_PLACEMENT, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.macros_num = macros_num
        self.nets_num = nets_num
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.macro_width_range = macro_width_range
        self.macro_height_range = macro_height_range
        self.net_degree_range = net_degree_range
        
        # Special Args for Clustered
        self.cluster_nums = cluster_nums
        self.cluster_std = cluster_std
        
        # Special Args for Mixed Size
        self.large_macro_ratio = large_macro_ratio
        self.large_macro_scale = large_macro_scale
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            MACRO_PLACEMENT_TYPE.UNIFORM: self._generate_uniform,
            MACRO_PLACEMENT_TYPE.CLUSTERED: self._generate_clustered,
            MACRO_PLACEMENT_TYPE.MIXED_SIZE: self._generate_mixed_size,
        }

    def _generate_macros(self, width_range: tuple, height_range: tuple, num: int) -> List[dict]:
        widths = np.random.uniform(width_range[0], width_range[1], size=num)
        heights = np.random.uniform(height_range[0], height_range[1], size=num)
        macros = []
        for i in range(num):
            macros.append({
                "width": float(widths[i]),
                "height": float(heights[i]),
            })
        return macros

    def _generate_nets(self, macros_num: int) -> List[dict]:
        nets = []
        for _ in range(self.nets_num):
            degree = np.random.randint(
                self.net_degree_range[0], self.net_degree_range[1] + 1
            )
            degree = min(degree, macros_num)
            macro_indices = np.random.choice(
                macros_num, size=degree, replace=False
            ).tolist()
            pin_offsets = {}
            for idx in macro_indices:
                pin_offsets[idx] = [
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.3, 0.3)
                ]
            nets.append({
                "macros": macro_indices,
                "pin_offsets": pin_offsets,
            })
        return nets

    def _generate_uniform(self) -> MacroPlacementTask:
        macros = self._generate_macros(
            self.macro_width_range, self.macro_height_range, self.macros_num
        )
        nets = self._generate_nets(self.macros_num)
        
        task_data = MacroPlacementTask(precision=self.precision)
        task_data.from_data(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            macros=macros,
            nets=nets,
        )
        return task_data

    def _generate_clustered(self) -> MacroPlacementTask:
        macros = self._generate_macros(
            self.macro_width_range, self.macro_height_range, self.macros_num
        )
        
        # Generate clustered nets: macros within a cluster share more nets
        macros_per_cluster = self.macros_num // self.cluster_nums
        cluster_assignments = []
        for c in range(self.cluster_nums):
            start = c * macros_per_cluster
            end = start + macros_per_cluster if c < self.cluster_nums - 1 else self.macros_num
            cluster_assignments.append(list(range(start, end)))
        
        nets = []
        for _ in range(self.nets_num):
            degree = np.random.randint(
                self.net_degree_range[0], self.net_degree_range[1] + 1
            )
            # With 70% probability, pick from same cluster
            if np.random.random() < 0.7:
                cluster_idx = np.random.randint(0, self.cluster_nums)
                pool = cluster_assignments[cluster_idx]
            else:
                pool = list(range(self.macros_num))
            
            degree = min(degree, len(pool))
            macro_indices = np.random.choice(pool, size=degree, replace=False).tolist()
            pin_offsets = {}
            for idx in macro_indices:
                pin_offsets[idx] = [
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.3, 0.3)
                ]
            nets.append({
                "macros": macro_indices,
                "pin_offsets": pin_offsets,
            })
        
        task_data = MacroPlacementTask(precision=self.precision)
        task_data.from_data(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            macros=macros,
            nets=nets,
        )
        return task_data

    def _generate_mixed_size(self) -> MacroPlacementTask:
        n_large = int(self.macros_num * self.large_macro_ratio)
        n_small = self.macros_num - n_large
        
        large_w_range = (
            self.macro_width_range[0] * self.large_macro_scale,
            self.macro_width_range[1] * self.large_macro_scale
        )
        large_h_range = (
            self.macro_height_range[0] * self.large_macro_scale,
            self.macro_height_range[1] * self.large_macro_scale
        )
        
        large_macros = self._generate_macros(large_w_range, large_h_range, n_large)
        small_macros = self._generate_macros(
            self.macro_width_range, self.macro_height_range, n_small
        )
        macros = large_macros + small_macros
        nets = self._generate_nets(self.macros_num)
        
        task_data = MacroPlacementTask(precision=self.precision)
        task_data.from_data(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            macros=macros,
            nets=nets,
        )
        return task_data
