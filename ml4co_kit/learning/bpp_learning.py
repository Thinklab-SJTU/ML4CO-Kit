r"""
BPP Learning Utilities.

包含：
- 特征提取
- 性能签名 / Label 计算（简化版）
- 一个简单的全连接 DNN，用于预测最佳 heuristic_id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml4co_kit.task.bpp.bpp_task import BPPTask
from ml4co_kit.solver.bpp_solver import BPPHeuristicSolver


# ----------- 特征提取 -----------

@dataclass
class BPPFeatures:
    values: np.ndarray  # 1D feature vector

    @staticmethod
    def from_task(task: BPPTask) -> "BPPFeatures":
        items = np.asarray(task.items, dtype=float)
        bins = np.asarray(task.bin_sizes, dtype=float)

        # Item features
        num_items = float(items.shape[0])
        sum_items = float(items.sum())
        min_item = float(items.min())
        max_item = float(items.max())
        mean_item = float(items.mean())
        std_item = float(items.std())
        var_item = float(items.var())

        # Bin features
        num_bins = float(bins.shape[0])
        min_bin = float(bins.min())
        max_bin = float(bins.max())
        mean_bin = float(bins.mean())
        std_bin = float(bins.std())
        var_bin = float(bins.var())

        # Cross features（这里做一个合理的近似）
        avg_fill_ratio = float((items / mean_bin).mean())
        avg_min_fill_ratio = float((items / min_bin).mean())
        avg_max_fill_ratio = float((items / max_bin).mean())

        feats = np.array(
            [
                num_items,
                sum_items,
                min_item,
                max_item,
                mean_item,
                std_item,
                var_item,
                num_bins,
                min_bin,
                max_bin,
                mean_bin,
                std_bin,
                var_bin,
                avg_fill_ratio,
                avg_min_fill_ratio,
                avg_max_fill_ratio,
            ],
            dtype=np.float32,
        )
        return BPPFeatures(values=feats)


# ----------- 性能签名 & label（简化） -----------

def profile_heuristics(task: BPPTask, delta: float = 0.01) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    对单个实例运行 8 种启发式，得到：
    - raw_costs: shape (8,)
    - performance_signature: shape (8,), 0/1 向量
    - label: 最终使用的 heuristic_id（这里取 signature 中第一个 1 的索引）

    这里用的是论文的 NP / SP 思路的简化版：
        NP = RP / ||RP||
        SP_i = 1 if NP_i <= NP_min + delta else 0
    """
    raw_costs = []
    tmp_task = BPPTask(precision=task.precision)
    tmp_task.from_data(items=task.items, bin_sizes=task.bin_sizes)

    for hid in range(8):
        solver = BPPHeuristicSolver(heuristic_id=hid)
        solver.solve(tmp_task)
        c = float(tmp_task.evaluate(tmp_task.sol))
        raw_costs.append(c)

    raw_costs = np.array(raw_costs, dtype=np.float32)

    # 归一化（对应论文 Eq. (1)，这里采用 L2 规范化）
    norm = float(np.linalg.norm(raw_costs)) or 1.0
    np_costs = raw_costs / norm
    np_min = float(np_costs.min())

    signature = (np_costs <= np_min + delta).astype(np.int64)
    # 简化：label = 第一个 1 的位置；若全 0，则取 argmin
    if signature.sum() == 0:
        label = int(np.argmin(raw_costs))
        signature[label] = 1
    else:
        label = int(np.where(signature == 1)[0][0])

    return raw_costs, signature, label


# ----------- DNN 模型结构（对应论文的深度全连接网络） -----------

class BPPHeuristicSelector(nn.Module):
    r"""
    一个简单的全连接网络，用于从 BPP 特征预测 heuristic_id。

    结构（与论文描述相近）：
    - 输入维度：16（上面的 BPPFeatures）
    - 5 层 hidden，每层 128 节点，softplus 激活
    - 输出：8 维，softmax
    """

    def __init__(self, in_dim: int = 16, num_classes: int = 8, hidden_dim: int = 128, num_hidden_layers: int = 5):
        super().__init__()
        layers: List[nn.Module] = []

        dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Softplus())
            dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim)
        return: logits, (batch, num_classes)
        """
        h = self.backbone(x)
        logits = self.head(h)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回预测的 heuristic_id（argmax）
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1)
