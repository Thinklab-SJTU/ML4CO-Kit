r"""
Generator for Standard Cell Placement instances.

References:
    [1] Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration 
        for Modern VLSI Placement", DAC 2019.
    [2] ISPD 2005/2006 Placement Contest Benchmarks.
    [3] Kahng et al., "ICCAD-2015 CAD Contest in Incremental Timing-driven 
        Placement and Benchmark Suite", ICCAD 2015.
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
from ml4co_kit.task.eda.standard_cell_placement import StandardCellPlacementTask
from ml4co_kit.generator.eda.base import EDAGeneratorBase


class STD_CELL_PLACEMENT_TYPE(str, Enum):
    """Distribution types for standard cell placement generation."""
    UNIFORM = "uniform"
    CLUSTERED = "clustered"


class StandardCellPlacementGenerator(EDAGeneratorBase):
    """Generator for Standard Cell Placement instances."""
    
    def __init__(
        self, 
        distribution_type: STD_CELL_PLACEMENT_TYPE = STD_CELL_PLACEMENT_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        std_cells_num: int = 200,
        nets_num: int = 300,
        canvas_width: float = 100.0,
        canvas_height: float = 100.0,
        cell_width_range: tuple = (0.5, 2.0),
        cell_height: float = 1.0,
        net_degree_range: tuple = (2, 6),
        fixed_macros_num: int = 5,
        macro_width_range: tuple = (10.0, 25.0),
        macro_height_range: tuple = (10.0, 25.0),
        # special args for clustered
        cluster_nums: int = 5,
    ):
        # Super Initialization
        super(StandardCellPlacementGenerator, self).__init__(
            task_type=TASK_TYPE.STANDARD_CELL_PLACEMENT, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.std_cells_num = std_cells_num
        self.nets_num = nets_num
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.cell_width_range = cell_width_range
        self.cell_height = cell_height
        self.net_degree_range = net_degree_range
        self.fixed_macros_num = fixed_macros_num
        self.macro_width_range = macro_width_range
        self.macro_height_range = macro_height_range
        
        # Special Args for Clustered
        self.cluster_nums = cluster_nums
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            STD_CELL_PLACEMENT_TYPE.UNIFORM: self._generate_uniform,
            STD_CELL_PLACEMENT_TYPE.CLUSTERED: self._generate_clustered,
        }

    def _generate_std_cells(self) -> List[dict]:
        widths = np.random.uniform(
            self.cell_width_range[0], self.cell_width_range[1], size=self.std_cells_num
        )
        std_cells = []
        for i in range(self.std_cells_num):
            std_cells.append({
                "width": float(widths[i]),
                "height": float(self.cell_height),
            })
        return std_cells

    def _generate_fixed_macros(self) -> List[dict]:
        fixed_macros = []
        for _ in range(self.fixed_macros_num):
            w = np.random.uniform(self.macro_width_range[0], self.macro_width_range[1])
            h = np.random.uniform(self.macro_height_range[0], self.macro_height_range[1])
            x = np.random.uniform(w / 2, self.canvas_width - w / 2)
            y = np.random.uniform(h / 2, self.canvas_height - h / 2)
            fixed_macros.append({
                "width": float(w),
                "height": float(h),
                "x": float(x),
                "y": float(y),
            })
        return fixed_macros

    def _generate_nets(self) -> List[dict]:
        nets = []
        for _ in range(self.nets_num):
            degree = np.random.randint(
                self.net_degree_range[0], self.net_degree_range[1] + 1
            )
            degree = min(degree, self.std_cells_num)
            cell_indices = np.random.choice(
                self.std_cells_num, size=degree, replace=False
            ).tolist()
            pin_offsets = {}
            for idx in cell_indices:
                pin_offsets[idx] = [
                    np.random.uniform(-0.2, 0.2),
                    np.random.uniform(-0.2, 0.2)
                ]
            nets.append({
                "cells": cell_indices,
                "pin_offsets": pin_offsets,
            })
        return nets

    def _generate_uniform(self) -> StandardCellPlacementTask:
        std_cells = self._generate_std_cells()
        fixed_macros = self._generate_fixed_macros()
        nets = self._generate_nets()
        
        task_data = StandardCellPlacementTask(precision=self.precision)
        task_data.from_data(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            std_cells=std_cells,
            nets=nets,
            fixed_macros=fixed_macros,
        )
        return task_data

    def _generate_clustered(self) -> StandardCellPlacementTask:
        std_cells = self._generate_std_cells()
        fixed_macros = self._generate_fixed_macros()
        
        # Assign cells to clusters, nets preferentially connect within cluster
        cells_per_cluster = self.std_cells_num // self.cluster_nums
        cluster_assignments = []
        for c in range(self.cluster_nums):
            start = c * cells_per_cluster
            end = start + cells_per_cluster if c < self.cluster_nums - 1 else self.std_cells_num
            cluster_assignments.append(list(range(start, end)))
        
        nets = []
        for _ in range(self.nets_num):
            degree = np.random.randint(
                self.net_degree_range[0], self.net_degree_range[1] + 1
            )
            if np.random.random() < 0.7:
                cluster_idx = np.random.randint(0, self.cluster_nums)
                pool = cluster_assignments[cluster_idx]
            else:
                pool = list(range(self.std_cells_num))
            
            degree = min(degree, len(pool))
            cell_indices = np.random.choice(pool, size=degree, replace=False).tolist()
            pin_offsets = {}
            for idx in cell_indices:
                pin_offsets[idx] = [
                    np.random.uniform(-0.2, 0.2),
                    np.random.uniform(-0.2, 0.2)
                ]
            nets.append({
                "cells": cell_indices,
                "pin_offsets": pin_offsets,
            })
        
        task_data = StandardCellPlacementTask(precision=self.precision)
        task_data.from_data(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
            std_cells=std_cells,
            nets=nets,
            fixed_macros=fixed_macros,
        )
        return task_data
