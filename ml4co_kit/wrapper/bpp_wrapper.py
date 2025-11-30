r"""
BPP Wrapper.

支持：
- 从 txt 加载 BPP 实例
- 批量生成 + 求解 + 写回 txt
"""

from __future__ import annotations

import pathlib
from typing import List

import numpy as np

from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.packing.bpp import BPPTask
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.utils.process_utils import tqdm_by_time


class BPPWrapper(WrapperBase):
    r"""
    BPP Wrapper.

    txt 格式约定为一行一个实例：

    items v1 v2 ... vN bins b1 b2 ... bM output a0 a1 ... aN-1 | t0 t1 ... tK-1

    其中：
    - items 后面是所有 item 尺寸
    - bins 后面是这一实例中给定的一批箱子容量（每个箱子最多使用一次）
    - output 部分：
        - a_i 是第 i 个 item 分配到的 bin 索引（0-based）
        - '|' 右侧 t_k 是第 k 个 bin 的类型索引（0-based）
      若没有解，可以省略 output 或写一个占位符（比如 -1）。
    """

    def __init__(
        self,
        precision: np.dtype = np.float32,
    ):
        super().__init__(task_type=TASK_TYPE.BPP, precision=precision)
        self.task_list: List[BPPTask] = []

    # ----------- 从 txt 读取 -----------
    def from_txt(
        self,
        file_path: pathlib.Path,
        ref: bool = False,
        overwrite: bool = True,
        show_time: bool = False,
    ):
        """Read task data from ``.txt`` file."""
        if overwrite:
            self.task_list = []

        with open(file_path, "r") as f:
            load_msg = f"Loading data from {file_path}"
            for _, line in tqdm_by_time(enumerate(f), load_msg, show_time):
                line = line.strip()
                if not line:
                    continue

                # 解析 items / bins / output
                # 格式：items ... bins ... output ...
                if "items " not in line or " bins " not in line:
                    raise ValueError(f"Invalid BPP line format: {line}")

                split_0 = line.split("items ")[1]
                split_1 = split_0.split(" bins ")
                items_str = split_1[0].strip()

                # 如果没有 output 部分，可以只包含 items / bins
                if " output " in split_1[1]:
                    split_2 = split_1[1].split(" output ")
                    bins_str = split_2[0].strip()
                    output_str = split_2[1].strip()
                else:
                    bins_str = split_1[1].strip()
                    output_str = ""

                # 解析 items
                items_vals = [float(v) for v in items_str.split(" ") if v != ""]
                items = np.array(items_vals, dtype=self.precision)

                # 解析 bins
                bin_vals = [float(v) for v in bins_str.split(" ") if v != ""]
                bin_sizes = np.array(bin_vals, dtype=self.precision)

                # 新建 task
                task = BPPTask(precision=self.precision)
                sol = None

                # 解析解
                if output_str:
                    if "|" in output_str:
                        left, right = output_str.split("|")
                        left = left.strip()
                        right = right.strip()
                        if left:
                            item_to_bin = np.array(
                                [int(x) for x in left.split(" ") if x != ""],
                                dtype=np.int64,
                            )
                        else:
                            item_to_bin = np.array([], dtype=np.int64)
                        if right:
                            bin_type_indices = np.array(
                                [int(x) for x in right.split(" ") if x != ""],
                                dtype=np.int64,
                            )
                        else:
                            bin_type_indices = np.array([], dtype=np.int64)
                        sol = (item_to_bin, bin_type_indices)
                    else:
                        # 只给了 item_to_bin，不给 bin_type_indices 的情况：不推荐，
                        # 这里简单忽略
                        pass

                task.from_data(items=items, bin_sizes=bin_sizes, sol=sol, ref=ref)
                self.task_list.append(task)

    # ----------- 写入 txt -----------
    def to_txt(
        self,
        file_path: pathlib.Path,
        show_time: bool = False,
        mode: str = "w",
    ):
        """Write task data to ``.txt`` file."""
        check_file_path(file_path)

        with open(file_path, mode) as f:
            write_msg = f"Writing data to {file_path}"
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                # 基本检查
                task._check_items_not_none()
                task._check_bin_sizes_not_none()

                items = task.items
                bin_sizes = task.bin_sizes
                sol = task.sol

                # items
                f.write("items ")
                f.write(" ".join(str(float(v)) for v in items))
                f.write(" bins ")
                f.write(" ".join(str(float(v)) for v in bin_sizes))

                # output
                f.write(" output ")
                if sol is not None:
                    item_to_bin, bin_type_indices = BPPTask._unpack_sol(sol)
                    left = " ".join(str(int(a)) for a in item_to_bin.tolist())
                    right = " ".join(str(int(t)) for t in bin_type_indices.tolist())
                    f.write(left)
                    f.write(" | ")
                    f.write(right)
                else:
                    f.write("")  # 无解就空着

                f.write("\n")
            f.close()

