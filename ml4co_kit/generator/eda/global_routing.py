r"""
Generator for Global Routing instances.

References:
    [1] ISPD 2007/2008 Global Routing Contest Benchmarks.
    [2] Wu et al., "NCTU-GR 2.0: Multithreaded Collision-Aware Global Routing 
        with Bounded-Length Maze Routing", TCAD 2013.
    [3] Liu et al., "Global Placement with Deep Learning-Enabled Explicit 
        Routability Optimization", DATE 2021.
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
from ml4co_kit.task.eda.routing import GlobalRoutingTask
from ml4co_kit.generator.eda.base import EDAGeneratorBase


class GLOBAL_ROUTING_TYPE(str, Enum):
    """Distribution types for global routing generation."""
    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    NON_UNIFORM_CAP = "non_uniform_cap"


class GlobalRoutingGenerator(EDAGeneratorBase):
    """Generator for Global Routing instances."""
    
    def __init__(
        self, 
        distribution_type: GLOBAL_ROUTING_TYPE = GLOBAL_ROUTING_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        grid_width: int = 32,
        grid_height: int = 32,
        nets_num: int = 100,
        net_degree_range: tuple = (2, 5),
        default_h_capacity: int = 10,
        default_v_capacity: int = 10,
        # special args for clustered
        cluster_nums: int = 4,
        cluster_std: float = 0.15,
        # special args for non_uniform_cap
        blockage_ratio: float = 0.1,
    ):
        # Super Initialization
        super(GlobalRoutingGenerator, self).__init__(
            task_type=TASK_TYPE.GLOBAL_ROUTING, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.nets_num = nets_num
        self.net_degree_range = net_degree_range
        self.default_h_capacity = default_h_capacity
        self.default_v_capacity = default_v_capacity
        
        # Special Args for Clustered
        self.cluster_nums = cluster_nums
        self.cluster_std = cluster_std
        
        # Special Args for Non-Uniform Capacity
        self.blockage_ratio = blockage_ratio
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            GLOBAL_ROUTING_TYPE.UNIFORM: self._generate_uniform,
            GLOBAL_ROUTING_TYPE.CLUSTERED: self._generate_clustered,
            GLOBAL_ROUTING_TYPE.NON_UNIFORM_CAP: self._generate_non_uniform_cap,
        }

    def _generate_nets_for_grid(self, pin_positions: np.ndarray = None) -> List[dict]:
        nets = []
        for _ in range(self.nets_num):
            degree = np.random.randint(
                self.net_degree_range[0], self.net_degree_range[1] + 1
            )
            if pin_positions is not None:
                indices = np.random.choice(len(pin_positions), size=degree, replace=False)
                pins = pin_positions[indices].tolist()
            else:
                pins = []
                for _ in range(degree):
                    gx = np.random.randint(0, self.grid_width)
                    gy = np.random.randint(0, self.grid_height)
                    pins.append([int(gx), int(gy)])
            nets.append({"pins": pins})
        return nets

    def _generate_uniform(self) -> GlobalRoutingTask:
        h_capacity = np.full(
            (self.grid_height, self.grid_width - 1), self.default_h_capacity, 
            dtype=self.precision
        )
        v_capacity = np.full(
            (self.grid_height - 1, self.grid_width), self.default_v_capacity, 
            dtype=self.precision
        )
        nets = self._generate_nets_for_grid()
        
        task_data = GlobalRoutingTask(precision=self.precision)
        task_data.from_data(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            h_capacity=h_capacity,
            v_capacity=v_capacity,
            nets=nets,
        )
        return task_data

    def _generate_clustered(self) -> GlobalRoutingTask:
        h_capacity = np.full(
            (self.grid_height, self.grid_width - 1), self.default_h_capacity, 
            dtype=self.precision
        )
        v_capacity = np.full(
            (self.grid_height - 1, self.grid_width), self.default_v_capacity, 
            dtype=self.precision
        )
        
        # Generate pin positions clustered around centers
        centers = np.random.uniform(
            low=[0, 0], high=[self.grid_width, self.grid_height],
            size=(self.cluster_nums, 2)
        )
        all_pins = []
        pins_per_net = self.nets_num * 3  # average 3 pins per net
        for _ in range(pins_per_net):
            center = centers[np.random.randint(0, self.cluster_nums)]
            gx = int(np.clip(
                np.random.normal(center[0], self.cluster_std * self.grid_width),
                0, self.grid_width - 1
            ))
            gy = int(np.clip(
                np.random.normal(center[1], self.cluster_std * self.grid_height),
                0, self.grid_height - 1
            ))
            all_pins.append([gx, gy])
        all_pins = np.array(all_pins)
        
        nets = self._generate_nets_for_grid(all_pins)
        
        task_data = GlobalRoutingTask(precision=self.precision)
        task_data.from_data(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            h_capacity=h_capacity,
            v_capacity=v_capacity,
            nets=nets,
        )
        return task_data

    def _generate_non_uniform_cap(self) -> GlobalRoutingTask:
        h_capacity = np.full(
            (self.grid_height, self.grid_width - 1), self.default_h_capacity, 
            dtype=self.precision
        )
        v_capacity = np.full(
            (self.grid_height - 1, self.grid_width), self.default_v_capacity, 
            dtype=self.precision
        )
        
        # Randomly reduce capacity (simulate blockages from macros/IPs)
        n_h_blockages = int(self.blockage_ratio * h_capacity.size)
        n_v_blockages = int(self.blockage_ratio * v_capacity.size)
        
        h_indices = np.random.choice(h_capacity.size, size=n_h_blockages, replace=False)
        h_capacity.flat[h_indices] = 0
        
        v_indices = np.random.choice(v_capacity.size, size=n_v_blockages, replace=False)
        v_capacity.flat[v_indices] = 0
        
        nets = self._generate_nets_for_grid()
        
        task_data = GlobalRoutingTask(precision=self.precision)
        task_data.from_data(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            h_capacity=h_capacity,
            v_capacity=v_capacity,
            nets=nets,
        )
        return task_data
