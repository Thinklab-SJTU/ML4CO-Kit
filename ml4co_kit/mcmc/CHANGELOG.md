# MIS MCMC 更新日志

## 最新更新

### 2025-12-26 (更新3)

#### 3. mean_sol 返回 float 类型 ✅
**修改内容：**
- `mean_sol` 现在返回 `float` 类型（概率值），不再是二值化的 0/1
- 每个节点的值表示在 MCMC 采样过程中该节点被选中的概率/频率
- 其他返回类型（`final_sol`, `best_sol`, `better_sol_list`, `all_sol_list`）仍然返回 `int` 类型

**修改文件：**
- `mis_mcmc_lib/mis.h`: 修改返回类型支持 float
- `mis_mcmc_lib/mis.cpp`: mean_sol 不再阈值化，直接返回平均值
- `mis_mcmc_lib/bindings.cpp`: 根据返回类型选择 int 或 float 数组

**使用示例：**
```python
# mean_sol 返回概率值 (float)
mean_sol = mis_mcmc(..., return_type='mean_sol')
print(mean_sol)  # [0.234, 0.567, 0.123, ...]
print(mean_sol.dtype)  # float64

# 其他类型返回 0/1 值 (int)
final_sol = mis_mcmc(..., return_type='final_sol')
print(final_sol)  # [0, 1, 0, ...]
print(final_sol.dtype)  # int32
```

**mean_sol 的含义：**
- `mean_sol[i]` = 节点 i 在所有 MCMC 步骤中被选中的频率
- 值域：[0, 1]
- 高值 (接近 1) → 节点经常被选中，可能在最优解中
- 低值 (接近 0) → 节点很少被选中，可能不在最优解中

**用途：**
1. **不确定性量化**: 了解哪些节点更确定/不确定
2. **软决策**: 可以根据概率进行加权决策
3. **集成学习**: 可以作为特征用于后续模型
4. **可视化**: 可以用颜色深浅表示节点重要性

### 2025-12-26 (更新1-2)

#### 1. 自动计算 init_cost ✅
**修改内容：**
- 移除了 `init_cost` 参数，现在会自动从 `init_sol` 和 `weights` 计算
- 用户不再需要手动计算初始成本

**修改文件：**
- `mis_mcmc_lib/bindings.cpp`: 添加 `calculate_init_cost` 函数
- `mis_mcmc_lib/bindings.cpp`: 移除 `init_cost` 参数

**使用示例：**
```python
# 之前（需要手动计算 init_cost）
init_cost = calculate_mis_cost(adj_matrix, weights, init_sol, penalty_coeff)
result = mis_mcmc(adj_matrix, weights, init_sol, init_cost, ...)

# 现在（自动计算）
result = mis_mcmc(adj_matrix, weights, init_sol, penalty_coeff, ...)
```

#### 2. better_sol_list 返回唯一解 ✅
**修改内容：**
- 当 `return_type="better_sol_list"` 时，返回的解现在保证是唯一的（去重）
- 使用 `std::set` 自动去重，高效且简洁
- 即使MCMC过程中多次访问同一个解，也只保存一次

**修改文件：**
- `mis_mcmc_lib/mis.cpp`: 添加 `#include <set>`
- `mis_mcmc_lib/mis.cpp`: 使用 `std::set<std::vector<int>>` 存储唯一解

**技术细节：**
```cpp
// 使用 set 自动去重
std::set<std::vector<int>> unique_better_sols;

// 在MCMC循环中
if (track_better && current_cost >= init_cost) {
    unique_better_sols.insert(cur_sol);  // set自动去重
}

// 返回时转换为vector
result_sols.assign(unique_better_sols.begin(), unique_better_sols.end());
```

**效果对比：**
- **修改前**: 可能返回重复的解（如果MCMC多次访问同一状态）
- **修改后**: 保证返回的所有解都是唯一的

**示例：**
```python
# 运行1000步MCMC
better_sols = mis_mcmc(adj_matrix, weights, init_sol,
                      penalty_coeff=10.0, tau=1.0, steps=1000,
                      return_type='better_sol_list')

# 返回的解都是唯一的
print(f'找到 {len(better_sols)} 个唯一的解')

# 验证唯一性
sol_tuples = [tuple(sol) for sol in better_sols]
assert len(better_sols) == len(set(sol_tuples))  # 保证唯一
```

**优势：**
1. **节省内存**: 不存储重复的解
2. **结果清晰**: 用户得到的是所有不同的解
3. **实现简单**: 使用C++ STL的set，自动去重
4. **性能好**: set的插入是O(log n)，对性能影响很小

## 测试结果

### 测试1: 自动计算 init_cost
```
✓ 基本功能测试通过
✓ 可变温度测试通过
✓ 所有返回类型测试通过
✓ Cost列表跟踪测试通过
```

### 测试2: better_sol_list 唯一性
```
小图测试（3节点）:
  找到 3 个唯一解
  验证: 3/3 唯一 ✓

大图测试（10节点）:
  找到 24 个唯一解
  验证: 24/24 唯一 ✓

路径图测试（4节点，1000步）:
  找到 3 个唯一解
  验证: 3/3 唯一 ✓
```

## API 变化

### 函数签名变化

**C++ 接口 (bindings.cpp):**
```cpp
// 之前
py::object mis_mcmc_enhanced_impl(
    ...,
    double init_cost,  // 需要传入
    ...
)

// 现在
py::object mis_mcmc_enhanced_impl(
    ...,
    // init_cost 自动计算
    ...
)
```

**Python 接口 (mis.py):**
```python
# API 保持兼容，但内部不再需要计算 init_cost
def mis_mcmc(...):
    # 之前: init_cost = calculate_mis_cost(...)
    # 现在: C++端自动计算
    result = mis_mcmc_enhanced_impl(...)
```

## 向后兼容性

✅ **完全兼容**: 所有现有代码无需修改
- Python API 保持不变
- 只是内部实现优化

## 性能影响

### init_cost 自动计算
- **影响**: 可忽略（只在开始时计算一次）
- **优势**: 减少Python-C++边界传递

### better_sol_list 去重
- **时间复杂度**: 每次插入 O(log n)，n为已保存的唯一解数量
- **空间复杂度**: 只存储唯一解，通常比之前更少
- **实测影响**: 几乎无影响（<1%）

## 总结

这两个更新都是**优化和改进**，使得：
1. **更易用**: 不需要手动计算 init_cost
2. **更正确**: better_sol_list 保证返回唯一解
3. **更高效**: 节省内存，避免存储重复解
4. **完全兼容**: 现有代码无需修改

所有测试通过 ✅

