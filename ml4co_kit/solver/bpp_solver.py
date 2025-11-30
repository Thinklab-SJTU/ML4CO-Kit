r"""
Heuristic solver for 1D Variable Sized Bin Packing (BPP).

实现8种启发式组合：
- Assignment: Best / First / Next / Worst Fit
- Allocation: Best Fit / Expect Fit
"""


from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import numpy as np

from ml4co_kit.task.base import TASK_TYPE, TaskBase
from ml4co_kit.task.packing.bpp import BPPTask
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class ASSIGNMENT_HEURISTIC(str, Enum):
    BEST_FIT = "best_fit"
    FIRST_FIT = "first_fit"
    NEXT_FIT = "next_fit"
    WORST_FIT = "worst_fit"


class ALLOCATION_HEURISTIC(str, Enum):
    BEST_FIT = "best_fit"
    EXPECT_FIT = "expect_fit"


# 8 种组合
HEURISTIC_SPACE: List[Tuple[ASSIGNMENT_HEURISTIC, ALLOCATION_HEURISTIC]] = [
    (ASSIGNMENT_HEURISTIC.BEST_FIT,  ALLOCATION_HEURISTIC.BEST_FIT),   # 0
    (ASSIGNMENT_HEURISTIC.BEST_FIT,  ALLOCATION_HEURISTIC.EXPECT_FIT), # 1
    (ASSIGNMENT_HEURISTIC.FIRST_FIT, ALLOCATION_HEURISTIC.BEST_FIT),   # 2
    (ASSIGNMENT_HEURISTIC.FIRST_FIT, ALLOCATION_HEURISTIC.EXPECT_FIT), # 3
    (ASSIGNMENT_HEURISTIC.NEXT_FIT,  ALLOCATION_HEURISTIC.BEST_FIT),   # 4
    (ASSIGNMENT_HEURISTIC.NEXT_FIT,  ALLOCATION_HEURISTIC.EXPECT_FIT), # 5
    (ASSIGNMENT_HEURISTIC.WORST_FIT, ALLOCATION_HEURISTIC.BEST_FIT),   # 6
    (ASSIGNMENT_HEURISTIC.WORST_FIT, ALLOCATION_HEURISTIC.EXPECT_FIT), # 7
]


class BPPHeuristicSolver(SolverBase):
    r"""
    BPP 启发式求解器，使用固定的 (assignment, allocation) 组合。
    可与深度学习模型结合，由模型预测 heuristic_id 再调用该 Solver。
    """

    def __init__(
        self,
        heuristic_id: int = 0,
    ):
        if heuristic_id < 0 or heuristic_id >= len(HEURISTIC_SPACE):
            raise ValueError("heuristic_id must be in [0, 7].")
        assignment, allocation = HEURISTIC_SPACE[heuristic_id]
        self.heuristic_id = heuristic_id
        self.assignment = assignment
        self.allocation = allocation

        super().__init__(solver_type=SOLVER_TYPE.DIY)

    # ----------- 核心求解函数 -----------
    def _solve(self, task_data: TaskBase):
        if task_data.task_type != TASK_TYPE.BPP:
            raise ValueError("BPPHeuristicSolver only supports TASK_TYPE.BPP.")
        assert isinstance(task_data, BPPTask)

        items = np.asarray(task_data.items, dtype=float)
        bin_sizes = np.asarray(task_data.bin_sizes, dtype=float)
        n_items = items.shape[0]
        n_types = bin_sizes.shape[0]

        # 初始化 opened bins：列表中每个元素为 dict
        # {'type_idx': int, 'remaining': float}
        opened_bins: List[dict] = []
        # item -> bin 映射
        item_to_bin = -np.ones(shape=(n_items,), dtype=np.int64)

        # 对 NEXT_FIT 需要一个“当前 bin 指针”
        current_bin_idx = None

        # items 按给定顺序处理（如需 FFD 可在外面先排序）
        remaining_item_indices = list(range(n_items))

        for idx in remaining_item_indices:
            size = float(items[idx])

            # 1) 尝试在已有打开的 bins 中做 assignment
            bin_idx = self._assign_item_to_open_bins(
                size=size,
                opened_bins=opened_bins,
                assignment=self.assignment,
                current_bin_idx=current_bin_idx,
            )

            # 2) 如果找不到合适的 open bin，则做 allocation
            if bin_idx is None:
                # 根据 allocation heuristic 选择一个 bin 类型并新开一个 bin
                type_idx = self._allocate_new_bin(
                    size=size,
                    items=items,
                    remaining_indices=[i for i in remaining_item_indices if i >= idx],
                    bin_sizes=bin_sizes,
                    allocation=self.allocation,
                )
                # 创建新 bin
                bin_idx = len(opened_bins)
                opened_bins.append(
                    {
                        "type_idx": type_idx,
                        "remaining": float(bin_sizes[type_idx]) - size,
                    }
                )
                # NEXT_FIT：当前 bin 即新开的 bin
                if self.assignment == ASSIGNMENT_HEURISTIC.NEXT_FIT:
                    current_bin_idx = bin_idx
            else:
                # 在已有 bin 中放下 item，更新 remaining
                opened_bins[bin_idx]["remaining"] -= size

            # 记录分配结果
            item_to_bin[idx] = bin_idx

            # NEXT_FIT：如果当前 bin 容量不足以装下任意后续 item，
            # 可以视为“关闭”，交给后续 allocation，新开 bin
            if self.assignment == ASSIGNMENT_HEURISTIC.NEXT_FIT:
                if opened_bins[bin_idx]["remaining"] <= 0:
                    current_bin_idx = None

        # 构造 bin_type_indices
        n_bins = len(opened_bins)
        bin_type_indices = np.zeros(shape=(n_bins,), dtype=np.int64)
        for k, info in enumerate(opened_bins):
            bin_type_indices[k] = info["type_idx"]

        # 写入解
        task_data.sol = (
            item_to_bin.astype(np.int64),
            bin_type_indices.astype(np.int64),
        )
        return task_data

    # ----------- Assignment 逻辑 -----------
    @staticmethod
    def _assign_item_to_open_bins(
        size: float,
        opened_bins: List[dict],
        assignment: ASSIGNMENT_HEURISTIC,
        current_bin_idx: int,
    ) -> int | None:
        """
        若能在已有打开的 bins 中找到合适的 bin，返回其索引，否则返回 None。
        """
        if not opened_bins:
            return None

        # 把剩余容量抽出来
        remaining = np.array([b["remaining"] for b in opened_bins], dtype=float)

        # 找出能装下该 item 的候选 bins
        candidates = np.where(remaining >= size)[0]
        if candidates.size == 0:
            return None

        if assignment == ASSIGNMENT_HEURISTIC.BEST_FIT:
            # 剩余空间最小的那个
            best = candidates[np.argmin(remaining[candidates] - size)]
            return int(best)

        elif assignment == ASSIGNMENT_HEURISTIC.FIRST_FIT:
            # 第一个能放的
            return int(candidates[0])

        elif assignment == ASSIGNMENT_HEURISTIC.NEXT_FIT:
            # 只尝试当前 bin
            if current_bin_idx is None:
                return None
            if remaining[current_bin_idx] >= size:
                return int(current_bin_idx)
            else:
                return None

        elif assignment == ASSIGNMENT_HEURISTIC.WORST_FIT:
            # 剩余空间最大的那个
            worst = candidates[np.argmax(remaining[candidates] - size)]
            return int(worst)

        else:
            raise NotImplementedError(f"Unknown assignment heuristic: {assignment}")

    # ----------- Allocation 逻辑 -----------
    @staticmethod
    def _allocate_new_bin(
        size: float,
        items: np.ndarray,
        remaining_indices: List[int],
        bin_sizes: np.ndarray,
        allocation: ALLOCATION_HEURISTIC,
    ) -> int:
        """
        选择要开哪种 bin 类型（返回 bin_type_idx）。
        - Best Fit: 选 capacity >= size 中，空余最小的。
        - Expect Fit: 看所有未装 items 的总量，选能刚好装下这些 items 的最小 bin，
          若总量大于所有 bin，则选最大的 bin。
        """
        feasible_types = np.where(bin_sizes >= size)[0]
        if feasible_types.size == 0:
            # 没有能装下当前 item 的 bin，退而求其次：选容量最大的 bin
            return int(np.argmax(bin_sizes))

        if allocation == ALLOCATION_HEURISTIC.BEST_FIT:
            waste = bin_sizes[feasible_types] - size
            best = feasible_types[np.argmin(waste)]
            return int(best)

        elif allocation == ALLOCATION_HEURISTIC.EXPECT_FIT:
            # 按论文描述：看剩余 items 的总和
            remaining_sum = float(items[remaining_indices].sum())
            # 找所有 capacity >= remaining_sum 的 bin 类型
            feasible_for_all = np.where(bin_sizes >= remaining_sum)[0]
            if feasible_for_all.size > 0:
                # 选容积最小的那个
                best = feasible_for_all[np.argmin(bin_sizes[feasible_for_all])]
                return int(best)
            else:
                # 否则，退回选最大容量的 bin
                return int(np.argmax(bin_sizes))

        else:
            raise NotImplementedError(f"Unknown allocation heuristic: {allocation}")
