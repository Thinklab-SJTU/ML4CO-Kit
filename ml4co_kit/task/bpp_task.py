r"""
1D Variable Sized Bin Packing Task.

- Items: 一个一维正数数组 T = {T_i}
- Bins: 候选 bin 类型 B = {B_j}（每种可无限使用）
- 目标：在满足容量约束下，最小化 cost：
    C = Σ_j ( 1 - load_j / capacity_j )
即每个使用的 bin 先有固定 cost 1，再扣掉“填充率”。
"""


from __future__ import annotations

import pathlib
from typing import Tuple, Optional

import numpy as np

from ml4co_kit.task.base import TaskBase, TASK_TYPE


class BPPTask(TaskBase):
    r"""
    1D Variable Sized Bin Packing Task.

    Attributes
    ----------
    items : np.ndarray
        一维数组，长度为 n_items，表示每个 item 的体积/重量（正数）。
    bin_sizes : np.ndarray
        一维数组，长度为 n_types，表示候选 bin 类型容量。
    sol :
        解的表示方式为一个二元组 (item_to_bin, bin_type_indices)：
        - item_to_bin: np.ndarray[int], shape (n_items,)
            第 i 个元素是该 item 被放入的 bin 索引 k（0 <= k < n_bins）
        - bin_type_indices: np.ndarray[int], shape (n_bins,)
            第 k 个元素是第 k 个 bin 使用的 bin 类型索引 t（0 <= t < n_types）

    注：这里假设每种 bin 类型数量无限。
    """

    def __init__(
        self,
        precision: np.dtype = np.float32,
    ):
        super().__init__(
            task_type=TASK_TYPE.BPP,
            minimize=True,                 # cost 越小越好
            precision=precision,
        )
        self.items: Optional[np.ndarray] = None
        self.bin_sizes: Optional[np.ndarray] = None

    # ----------- 基础检查 -----------
    def _check_items_dim(self):
        if self.items.ndim != 1:
            raise ValueError("``items`` should be a 1D array.")

    def _check_bin_sizes_dim(self):
        if self.bin_sizes.ndim != 1:
            raise ValueError("``bin_sizes`` should be a 1D array.")

    def _check_items_not_none(self):
        if self.items is None:
            raise ValueError("``items`` cannot be None!")

    def _check_bin_sizes_not_none(self):
        if self.bin_sizes is None:
            raise ValueError("``bin_sizes`` cannot be None!")

    # ----------- from_data 接口 -----------
    def from_data(
        self,
        items: np.ndarray = None,
        bin_sizes: np.ndarray = None,
        sol=None,
        ref: bool = False,
        name: str = None,
    ):
        """
        通过原始数据构造 BPP 实例。

        Parameters
        ----------
        items : np.ndarray
            1D array of item sizes (> 0).
        bin_sizes : np.ndarray
            1D array of candidate bin capacities (> 0).
        sol :
            若不为 None，则视为当前解：
            - 若 ref=True，则写入 ref_sol
            - 否则写入 sol
        ref : bool
            是否把 sol 视为 reference solution.
        name : str
            实例名称（可选）。
        """
        if items is not None:
            items = np.asarray(items, dtype=self.precision)
            if np.any(items <= 0):
                raise ValueError("All items must be positive.")
            self.items = items
            self._check_items_dim()

        if bin_sizes is not None:
            bin_sizes = np.asarray(bin_sizes, dtype=self.precision)
            if np.any(bin_sizes <= 0):
                raise ValueError("All bin sizes must be positive.")
            self.bin_sizes = bin_sizes
            self._check_bin_sizes_dim()

        if name is not None:
            self.name = name

        if sol is not None:
            if ref:
                self.ref_sol = sol
            else:
                self.sol = sol

    # ----------- 约束检查与 cost 计算 -----------
    @staticmethod
    def _unpack_sol(sol) -> Tuple[np.ndarray, np.ndarray]:
        """
        统一解的内部表示： (item_to_bin, bin_type_indices)
        """
        if not isinstance(sol, (tuple, list)) or len(sol) != 2:
            raise ValueError(
                "Solution for BPPTask must be a tuple (item_to_bin, bin_type_indices)."
            )
        item_to_bin, bin_type_indices = sol
        item_to_bin = np.asarray(item_to_bin, dtype=np.int64)
        bin_type_indices = np.asarray(bin_type_indices, dtype=np.int64)
        return item_to_bin, bin_type_indices

    def check_constraints(self, sol) -> bool:
        """
        检查容量约束是否满足。
        """
        self._check_items_not_none()
        self._check_bin_sizes_not_none()

        item_to_bin, bin_type_indices = self._unpack_sol(sol)
        n_items = self.items.shape[0]
        if item_to_bin.shape[0] != n_items:
            raise ValueError("Length of ``item_to_bin`` must equal number of items.")

        n_bins = bin_type_indices.shape[0]
        if n_bins == 0:
            # 空解视为非法
            return False

        # 每个 bin 类型索引必须在合法范围内
        if np.any((bin_type_indices < 0) | (bin_type_indices >= self.bin_sizes.shape[0])):
            return False

        # 逐 bin 计算负载并检查是否超过容量
        for k in range(n_bins):
            type_idx = bin_type_indices[k]
            capacity = float(self.bin_sizes[type_idx])
            # 找出所有分配到 bin k 的 item
            mask = (item_to_bin == k)
            load = float(self.items[mask].sum()) if np.any(mask) else 0.0
            if load - capacity > 1e-8:   # 容量约束
                return False

        return True

    def evaluate(self, sol) -> np.floating:
        """
        按论文的 cost 函数计算：
            C = Σ_j ( 1 - load_j / capacity_j )

        若解不满足约束，抛出异常。
        """
        if not self.check_constraints(sol):
            raise ValueError("Invalid BPP solution (violates capacity constraints).")

        item_to_bin, bin_type_indices = self._unpack_sol(sol)
        n_bins = bin_type_indices.shape[0]

        costs = []
        for k in range(n_bins):
            type_idx = bin_type_indices[k]
            capacity = float(self.bin_sizes[type_idx])
            mask = (item_to_bin == k)
            load = float(self.items[mask].sum()) if np.any(mask) else 0.0
            # waste cost of this bin
            ratio = load / capacity if capacity > 0 else 1.0
            costs.append(1.0 - ratio)

        total_cost = np.array(sum(costs), dtype=self.precision)
        return total_cost

    # ----------- 可视化（简单版本，可后续增强） -----------
    def render(
        self,
        save_path: pathlib.Path,
        with_sol: bool = True,
        figsize: tuple = (8, 4),
    ):
        """
        简单渲染：画出每个 bin 的填充率条形图。
        """
        import matplotlib.pyplot as plt

        self._check_items_not_none()
        self._check_bin_sizes_not_none()

        if not with_sol or self.sol is None:
            raise NotImplementedError("Please set a solution and with_sol=True.")

        item_to_bin, bin_type_indices = self._unpack_sol(self.sol)
        n_bins = bin_type_indices.shape[0]

        loads = []
        capacities = []
        for k in range(n_bins):
            type_idx = bin_type_indices[k]
            capacity = float(self.bin_sizes[type_idx])
            mask = (item_to_bin == k)
            load = float(self.items[mask].sum()) if np.any(mask) else 0.0
            loads.append(load)
            capacities.append(capacity)

        loads = np.array(loads)
        capacities = np.array(capacities)
        ratios = loads / capacities

        plt.figure(figsize=figsize)
        x = np.arange(n_bins)
        plt.bar(x, ratios)
        plt.ylim(0, 1.05)
        plt.xlabel("Bin Index")
        plt.ylabel("Fill Ratio")
        plt.title(f"BPP Solution (instance: {self.name})")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
