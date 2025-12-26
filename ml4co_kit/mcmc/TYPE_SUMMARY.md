# MIS MCMC 返回类型总结

## 返回类型对照表

| return_type | 返回数据类型 | 返回内容 | 值域 | 说明 |
|------------|------------|---------|------|------|
| `final_sol` | `np.ndarray[int32]` | 单个解 | {0, 1} | MCMC 最后一步的解 |
| `best_sol` | `np.ndarray[int32]` | 单个解 | {0, 1} | MCMC 过程中遇到的最优解 |
| `mean_sol` | `np.ndarray[float64]` | 单个解 | [0, 1] | 每个节点被选中的平均概率 |
| `better_sol_list` | `List[np.ndarray[int32]]` | 解列表 | {0, 1} | 所有 cost ≥ init_cost 的唯一解 |
| `all_sol_list` | `List[np.ndarray[int32]]` | 解列表 | {0, 1} | 所有 MCMC 步骤的解 |

## 详细说明

### 1. final_sol
- **类型**: `np.ndarray[int32]`
- **形状**: `(n,)` 其中 n 是节点数
- **值**: 0 或 1
- **含义**: MCMC 运行结束时的最终解

```python
final_sol = mis_mcmc(..., return_type='final_sol')
# 例如: [0, 1, 0, 1, 0]
```

### 2. best_sol
- **类型**: `np.ndarray[int32]`
- **形状**: `(n,)`
- **值**: 0 或 1
- **含义**: MCMC 过程中遇到的成本最高的解

```python
best_sol = mis_mcmc(..., return_type='best_sol')
# 例如: [1, 0, 0, 1, 0]
```

### 3. mean_sol ⭐ (特殊)
- **类型**: `np.ndarray[float64]`
- **形状**: `(n,)`
- **值**: 0.0 到 1.0 之间的浮点数
- **含义**: 每个节点在所有 MCMC 步骤中被选中的频率/概率

```python
mean_sol = mis_mcmc(..., return_type='mean_sol', steps=1000)
# 例如: [0.234, 0.567, 0.123, 0.789, 0.012]
# 解释:
#   节点 0: 在 1000 步中被选中了 234 次 (23.4%)
#   节点 1: 在 1000 步中被选中了 567 次 (56.7%)
#   ...
```

**mean_sol 的用途:**
- **不确定性量化**: 值接近 0.5 表示不确定，接近 0 或 1 表示确定
- **软决策**: 可以用概率值进行加权决策
- **特征提取**: 可以作为机器学习的特征
- **可视化**: 用颜色深浅表示节点重要性

### 4. better_sol_list
- **类型**: `List[np.ndarray[int32]]`
- **形状**: 列表长度可变，每个元素形状为 `(n,)`
- **值**: 每个解是 0 或 1
- **含义**: 所有成本 ≥ 初始成本的**唯一**解
- **特点**: 自动去重，不包含重复的解

```python
better_sols = mis_mcmc(..., return_type='better_sol_list')
# 例如: [
#   [1, 0, 1, 0, 0],
#   [0, 1, 0, 1, 0],
#   [1, 0, 0, 1, 0],
# ]
# 所有解都是唯一的（没有重复）
```

### 5. all_sol_list
- **类型**: `List[np.ndarray[int32]]`
- **形状**: 列表长度 = steps，每个元素形状为 `(n,)`
- **值**: 每个解是 0 或 1
- **含义**: 每一步 MCMC 的解
- **注意**: 可能包含重复的解

```python
all_sols = mis_mcmc(..., return_type='all_sol_list', steps=100)
# 长度为 100，包含每一步的解
# 可能有重复
```

## 类型检查示例

```python
import numpy as np
from ml4co_kit.mcmc import mis_mcmc

# 检查返回类型
final_sol = mis_mcmc(..., return_type='final_sol')
assert final_sol.dtype == np.int32
assert final_sol.ndim == 1
assert np.all((final_sol == 0) | (final_sol == 1))

mean_sol = mis_mcmc(..., return_type='mean_sol')
assert mean_sol.dtype == np.float64
assert mean_sol.ndim == 1
assert np.all((mean_sol >= 0) & (mean_sol <= 1))

better_sols = mis_mcmc(..., return_type='better_sol_list')
assert isinstance(better_sols, list)
assert all(sol.dtype == np.int32 for sol in better_sols)
```

## 性能对比

| return_type | 内存使用 | 计算开销 | 适用场景 |
|------------|---------|---------|---------|
| `final_sol` | 最小 | 最小 | 只需要最终结果 |
| `best_sol` | 小 | 小 | 需要最优解 |
| `mean_sol` | 中等 | 中等 | 需要概率信息 |
| `better_sol_list` | 中等 | 中等 | 需要多个好解（去重） |
| `all_sol_list` | 最大 | 小 | 需要完整轨迹 |

## 选择建议

1. **只需要一个解**: 使用 `final_sol` 或 `best_sol`
2. **需要概率/不确定性信息**: 使用 `mean_sol`
3. **需要多个不同的好解**: 使用 `better_sol_list`
4. **需要分析 MCMC 轨迹**: 使用 `all_sol_list`
5. **需要追踪成本变化**: 设置 `return_cost_list=True`

## 常见问题

### Q: 为什么 mean_sol 是 float 而不是 int？
A: mean_sol 表示概率/频率，不是具体的解。如果需要二值化，可以手动阈值化：
```python
mean_sol = mis_mcmc(..., return_type='mean_sol')
binary_sol = (mean_sol >= 0.5).astype(int)
```

### Q: better_sol_list 一定包含初始解吗？
A: 是的，如果初始解满足条件（cost >= init_cost），会被包含。

### Q: all_sol_list 会很大吗？
A: 是的，如果 steps 很大（如 10000），会占用较多内存。建议只在需要时使用。

### Q: 如何验证 mean_sol 的正确性？
A: 可以用 all_sol_list 验证：
```python
all_sols = mis_mcmc(..., return_type='all_sol_list')
mean_from_all = np.mean(all_sols, axis=0)
mean_sol = mis_mcmc(..., return_type='mean_sol')
# mean_from_all 应该接近 mean_sol
```
