r"""
Generator for 1D Variable Sized Bin Packing instances.
"""
from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np

from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.packing.bpp import BPPTask
from ml4co_kit.generator.base import GeneratorBase


class BPP_TYPE(str, Enum):
    """BPP 分布类型（可按需扩展）"""
    UNIFORM = "uniform"          # items 与 bin_sizes 都均匀分布
    HEAVY_TAILED = "heavy_tailed"  # items 偏重尾分布（举例）


class BPPGenerator(GeneratorBase):
    r"""
    Generator for BPP instances.

    Parameters
    ----------
    distribution_type : BPP_TYPE
    precision : np.dtype
    num_items : int
        item 个数
    num_bin_types : int
        bin 类型个数
    item_low, item_high : float
        item 尺寸采样区间
    bin_low, bin_high : float
        bin 容量采样区间
    """

    def __init__(
        self,
        distribution_type: BPP_TYPE = BPP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        num_items: int = 20,
        num_bin_types: int = 4,
        item_low: float = 0.1,
        item_high: float = 1.0,
        bin_low: float = 0.5,
        bin_high: float = 2.0,
    ):
        super().__init__(
            task_type=TASK_TYPE.BPP,
            distribution_type=distribution_type,
            precision=precision,
        )
        self.num_items = num_items
        self.num_bin_types = num_bin_types
        self.item_low = item_low
        self.item_high = item_high
        self.bin_low = bin_low
        self.bin_high = bin_high

        self.generate_func_dict = {
            BPP_TYPE.UNIFORM: self._generate_uniform,
            BPP_TYPE.HEAVY_TAILED: self._generate_heavy_tailed,
        }

    # ----------- 各种分布 -----------
    def _generate_uniform(self) -> BPPTask:
        items = np.random.uniform(
            low=self.item_low, high=self.item_high, size=(self.num_items,)
        ).astype(self.precision)
        bin_sizes = np.random.uniform(
            low=self.bin_low, high=self.bin_high, size=(self.num_bin_types,)
        ).astype(self.precision)

        data = BPPTask(precision=self.precision)
        data.from_data(items=items, bin_sizes=bin_sizes)
        return data

    def _generate_heavy_tailed(self) -> BPPTask:
        # 示例：items 使用对数正态分布，bin 仍然均匀
        items = np.random.lognormal(
            mean=0.0, sigma=0.75, size=(self.num_items,)
        ).astype(self.precision)
        # 归一化到 [item_low, item_high]
        items = items / items.max()
        items = self.item_low + (self.item_high - self.item_low) * items

        bin_sizes = np.random.uniform(
            low=self.bin_low, high=self.bin_high, size=(self.num_bin_types,)
        ).astype(self.precision)

        data = BPPTask(precision=self.precision)
        data.from_data(items=items, bin_sizes=bin_sizes)
        return data
