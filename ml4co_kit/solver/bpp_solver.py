r"""
Heuristic solver for 1D Variable Sized Bin Packing (BPP).

实现 8 种启发式组合：
- Assignment: Best / First / Next / Worst Fit
- Allocation: Best Fit / Expect Fit

本版本采用“有限箱子”模式：
- bin_sizes: 一批具体箱子，每个箱子最多使用一次。
"""
from __future__ import annotations

from enum import Enum
from typing import List, Tuple, Set

import numpy as np

from ml4co_kit.task.base import TASK_TYPE, TaskBase
from ml4co_kit.task.bpp.bpp_task import BPPTask
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

    【现在的语义】
    - bin_sizes 是一批具体箱子，每个箱子最多使用一次；
    - 求解过程中，我们维护：
        - available_bins: 还没被使用过的箱子索引集合；
        - opened_bins: 已经打开并分配过物品的箱子列表，每个元素记录：
            {"bin_id": j, "remaining": 剩余容量}.
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
        n_total_bins = bin_sizes.shape[0]

        # opened_bins: 列表中每个元素为 dict
        # {"bin_id": j, "remaining": float}
        opened_bins: List[dict] = []
        # item -> bin 索引（这是“第几个打开的箱子 k”，不是 bin_sizes 的索引）
        item_to_bin = -np.ones(shape=(n_items,), dtype=np.int64)

        # 可用箱子集合：bin_sizes 的索引 j
        available_bins: Set[int] = set(range(n_total_bins))

        # 对 NEXT_FIT 需要一个“当前 bin 指针”（指向 opened_bins 的索引）
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

            # 2) 如果找不到合适的 open bin，则做 allocation：从“剩余箱子”中选一个新箱子
            if bin_idx is None:
                bin_id = self._allocate_new_bin(
                    size=size,
                    items=items,
                    remaining_indices=[i for i in remaining_item_indices if i >= idx],
                    bin_sizes=bin_sizes,
                    available_bins=available_bins,
                    allocation=self.allocation,
                )
                if bin_id is None:
                    # 理论上不应该发生（因为我们生成数据时保证“箱子够用”），
                    # 如果发生，就直接抛异常，方便调试。
                    raise RuntimeError("No available bin can be allocated for item size "
                                       f"{size}, available_bins={available_bins}")

                # 标记该箱子已被占用
                if bin_id not in available_bins:
                    raise RuntimeError(f"Bin {bin_id} allocated twice.")
                available_bins.remove(bin_id)

                # 创建新 bin（在 opened_bins 中的位置）
                bin_idx = len(opened_bins)
                opened_bins.append(
                    {
                        "bin_id": int(bin_id),
                        "remaining": float(bin_sizes[bin_id]) - size,
                    }
                )
                # NEXT_FIT：当前 bin 即新开的 bin
                if self.assignment == ASSIGNMENT_HEURISTIC.NEXT_FIT:
                    current_bin_idx = bin_idx
            else:
                # 在已有 bin 中放置
                opened_bins[bin_idx]["remaining"] -= size
                if self.assignment == ASSIGNMENT_HEURISTIC.NEXT_FIT:
                    current_bin_idx = bin_idx

            # 记录该 item 属于第几个 opened bin（k）
            item_to_bin[idx] = bin_idx

        # 构造 bin_type_indices / solution：
        # - k: 第 k 个使用的 bin
        # - opened_bins[k]["bin_id"]: 对应 bin_sizes 中的哪一个箱子索引 j
        n_used_bins = len(opened_bins)
        bin_type_indices = np.zeros(shape=(n_used_bins,), dtype=np.int64)
        for k, info in enumerate(opened_bins):
            bin_type_indices[k] = info["bin_id"]

        sol = (item_to_bin.astype(np.int64), bin_type_indices.astype(np.int64))
        task_data.sol = sol
        return sol

    # ----------- Assignment 逻辑：在已经打开的 bin 中选一个放 -----------
    @staticmethod
    def _assign_item_to_open_bins(
        size: float,
        opened_bins: List[dict],
        assignment: ASSIGNMENT_HEURISTIC,
        current_bin_idx: int,
    ):
        """
        在已打开的 bins 里根据 assignment 策略选一个能放下 size 的 bin。

        返回值：
        - 若找到：返回 opened_bins 的索引 k
        - 若找不到：返回 None
        """
        if not opened_bins:
            return None

        remaining = np.array([b["remaining"] for b in opened_bins], dtype=float)

        # 找出能装下该 item 的候选 bins（按 opened_bins 的索引）
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

    # ----------- Allocation 逻辑：从剩余箱子集合里选一个新箱子 -----------
    @staticmethod
    def _allocate_new_bin(
        size: float,
        items: np.ndarray,
        remaining_indices: List[int],
        bin_sizes: np.ndarray,
        available_bins: Set[int],
        allocation: ALLOCATION_HEURISTIC,
    ) -> int | None:
        """
        选择要开哪个“具体箱子”（bin_id，是 bin_sizes 的索引）。

        只在 available_bins 这一子集里挑选。

        - Best Fit:
            在所有 capacity >= size 的候选中，选空余最小的。
        - Expect Fit:
            看剩余 items 的总量 remaining_sum：
            * 若存在 bin 容量 >= remaining_sum，则选其中容量最小的；
            * 否则，选容量最大的那个可行箱子。
        """
        if not available_bins:
            return None

        candidate_ids = np.array(sorted(available_bins), dtype=np.int64)
        candidate_caps = bin_sizes[candidate_ids]

        # 能装下当前物品的箱子
        feasible_mask = candidate_caps >= size
        if not np.any(feasible_mask):
            # 没有任何箱子能装下当前 item，这个实例基本无解；
            # 为了不悄悄出错，这里返回 None，让上层抛异常。
            return None

        feasible_ids = candidate_ids[feasible_mask]
        feasible_caps = bin_sizes[feasible_ids]

        if allocation == ALLOCATION_HEURISTIC.BEST_FIT:
            waste = feasible_caps - size
            best_idx = int(np.argmin(waste))
            return int(feasible_ids[best_idx])

        elif allocation == ALLOCATION_HEURISTIC.EXPECT_FIT:
            remaining_sum = float(items[remaining_indices].sum())
            # 能装下“所有剩余 items”的箱子
            feasible_all_mask = feasible_caps >= remaining_sum
            if np.any(feasible_all_mask):
                cand_ids = feasible_ids[feasible_all_mask]
                cand_caps = bin_sizes[cand_ids]
                # 选容量最小的那个
                best_idx = int(np.argmin(cand_caps))
                return int(cand_ids[best_idx])
            else:
                # 否则在 feasible_ids 中选容量最大的那个
                best_idx = int(np.argmax(feasible_caps))
                return int(feasible_ids[best_idx])

        else:
            raise NotImplementedError(f"Unknown allocation heuristic: {allocation}")
