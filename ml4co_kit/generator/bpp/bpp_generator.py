r"""
Generator for 1D Variable Sized Bin Packing instances.

采用“有限箱子”模式：
- items: 一批物品
- bin_sizes: 一批具体箱子，每个箱子只能用一次
"""
from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np

from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.bpp.bpp_task import BPPTask
from ml4co_kit.generator.base import GeneratorBase


class BPP_TYPE(str, Enum):
    """BPP 分布类型（可按需扩展）"""
    UNIFORM = "uniform"           # items 与 bin_sizes 都均匀分布
    HEAVY_TAILED = "heavy_tailed" # items 偏重尾分布（举例）


class BPPGenerator(GeneratorBase):
    r"""
    Generator for BPP instances（有限箱子版本）.

    Parameters
    ----------
    distribution_type : BPP_TYPE
    precision : np.dtype
    num_items : int
        item 个数
    num_bin_types : int
        【注意：这里的 num_bin_types 实际表示“箱子总个数 M”】
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
        num_bin_types: int = 10,   # 实际是“箱子总数 M”
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
        # 为了不改外部调用接口，这里名字仍叫 num_bin_types，
        # 但语义已经变成：一共有多少个具体箱子 M
        self.num_bins = num_bin_types
        self.item_low = item_low
        self.item_high = item_high
        self.bin_low = bin_low
        self.bin_high = bin_high

        self.generate_func_dict = {
            BPP_TYPE.UNIFORM: self._generate_uniform,
            BPP_TYPE.HEAVY_TAILED: self._generate_heavy_tailed,
        }

    # ----------- 内部工具：根据 items 生成“足够用的一批箱子” -----------
    def _sample_bins_for_items(self, items: np.ndarray) -> np.ndarray:
        """
        给定 items，反复采样，直到得到一批“够用”的 bin_sizes：

        1) 每个 item 至少能找到一个箱子装：
           max(bin_sizes) >= max(items)
        2) 总容量足够：
           sum(bin_sizes) >= sum(items)
        """
        total_items = float(items.sum())
        max_item = float(items.max())

        while True:
            bin_sizes = np.random.uniform(
                low=self.bin_low,
                high=self.bin_high,
                size=(self.num_bins,),
            ).astype(self.precision)

            if float(bin_sizes.max()) < max_item:
                # 没有箱子能装下最大 item，重新采
                continue

            if float(bin_sizes.sum()) < total_items:
                # 总容量不够，重新采
                continue

            return bin_sizes

    # ----------- 各种分布 -----------
    def _generate_uniform(self) -> BPPTask:
        # items 均匀分布
        items = np.random.uniform(
            low=self.item_low,
            high=self.item_high,
            size=(self.num_items,),
        ).astype(self.precision)

        # 为这批 items 生成一批“具体箱子”，每个箱子只能用一次
        bin_sizes = self._sample_bins_for_items(items)

        data = BPPTask(precision=self.precision)
        data.from_data(items=items, bin_sizes=bin_sizes)
        return data

    def _generate_heavy_tailed(self) -> BPPTask:
        # 示例：items 使用对数正态分布
        items = np.random.lognormal(
            mean=0.0,
            sigma=0.75,
            size=(self.num_items,),
        ).astype(self.precision)

        # 归一化到 [item_low, item_high]
        items = items / items.max()
        items = self.item_low + (self.item_high - self.item_low) * items
        items = items.astype(self.precision)

        # 同样根据 items 生成一批“足够用”的箱子
        bin_sizes = self._sample_bins_for_items(items)

        data = BPPTask(precision=self.precision)
        data.from_data(items=items, bin_sizes=bin_sizes)
        return data
