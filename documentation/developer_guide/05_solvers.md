# 05 Solver 模块开发者参考

*Updated: 2026-04-23.*

Solver 与 trainer 同层。它们消耗已累积的统计量，返回参数更新或闭合解。
**不拥有** MPI 执行、帧提取或 cache lifetime。

当前 solver 接口：

| 文件 | 职责 |
|---|---|
| `solvers/base.py` | `BaseSolver`，通用单次求解接口 |
| `solvers/fm_matrix.py` | `FMMatrixSolver`，从标准 FM 统计做闭合求解 |

---

## 层边界

```
topology + compute
  → Forcefield、reducer 输出（来自 run_post pickle）

solver
  → 消耗一个 batch，求解，返回 dict

workflow / trainer
  → 决定何时调用 solver、把求解结果发送到哪里
```

**Solver 应该拥有**：

- 求解配置
- 一个私有 `Forcefield` 副本
- 矩阵代数

**Solver 不应该拥有**：

- trajectory I/O
- MPI engine 创建
- 帧选择策略
- runtime cache
- 训练循环控制

---

## `BaseSolver`（solvers/base.py）

### 职责

| 方法 | 含义 |
|---|---|
| `__init__(forcefield, logger=None)` | deep-copy 并拥有一个 `Forcefield` |
| `schema()` | 描述 input/output dict 合同 |
| `get_params()` | 返回当前全参数向量 |
| `update_forcefield(params)` | 将全参数向量写回拥有的 `Forcefield` |
| `solve(batch)` | 抽象的单次求解入口 |

**新 solver 的设计规则**：

- 接受一个规范 batch dict
- 返回一个 plain result dict
- 在 shape 或合同不匹配时 fail-fast
- 保持 workflow / training 关切在 solver 代码之外

---

## `FMMatrixSolver`（solvers/fm_matrix.py）

`FMMatrixSolver` 是规范的闭合形式 FM 求解器。
消耗 `step_mode="fm"` 的 `run_post()` pickle 输出，一次调用完成求解。

### 支持模式

| Mode | 含义 |
|---|---|
| `ols` | 普通最小二乘 |
| `ridge` | Tikhonov 正则化求解 |
| `bayesian` | 对角 ARD Bayesian evidence 更新 |

### 构建

```python
from AceCG.solvers.fm_matrix import FMMatrixSolver

solver = FMMatrixSolver(
    trainer.forcefield,
    mode="ridge",
    ridge_alpha=1.0e-6,
)
```

Solver 拥有独立的 `Forcefield` 副本。更新 solver 参数不 mutate 调用者的 forcefield，
直到 workflow 选择传播结果。

---

## FM solver batch 合同

`FMMatrixSolver.solve(batch)` 期望标准 FM payload（来自 `run_post` pickle）：

| Key | Shape | 含义 |
|---|---:|---|
| `JtJ` | `(p, p)` | 归一化 FM 法矩阵 |
| `Jty` | `(p,)` | 归一化 FM 右端向量 |
| `y_sumsq` | scalar | 归一化目标力范数平方 |
| `nframe` | scalar | 贡献帧数 |
| `weight_sum` | scalar | 归一化前的总帧权重 |
| `n_atoms_obs` | scalar | 每帧被观测原子数 |

可选传递字段：

| Key | 含义 |
|---|---|
| `step_index` | 存在时复制进 `result["meta"]` |

### 结果合同

| Key | 含义 |
|---|---|
| `params` | 求解后的全参数向量 |
| `loss` | 求解参数处的归一化 FM loss |
| `mode` | 求解模式 |
| `meta` | 诊断信息（active 参数数、Bayesian 统计等）|

---

## FM 统计的科学合同

FM reducer 返回归一化二次统计量：

$$JtJ = \sum_i w_i J_i^T J_i,\quad Jty = \sum_i w_i J_i^T y_i,\quad y\_\text{sumsq} = \sum_i w_i y_i^T y_i$$

其中归一化权重满足 $\sum_i w_i = 1$。

Solver 直接使用：

$$L(\theta) = \frac{1}{2}\left(\theta^T JtJ\, \theta - 2\,\theta^T Jty + y\_\text{sumsq}\right)$$

Bayesian 模式下用 `weight_sum` 反推未归一化系统：

$$JtJ_{\text{raw}} = \text{weight\_sum} \cdot JtJ,\quad Jty_{\text{raw}} = \text{weight\_sum} \cdot Jty$$

有效标量观测数：

$$N_{\text{obs}} = 3 \cdot n\_\text{atoms\_obs} \cdot \text{weight\_sum}$$

这就是 FM payload 中必须保留 `weight_sum` 和 `n_atoms_obs` 的原因。

---

## Mask 语义

这是最重要的开发者合同。

### 规则 1：计算全 FM 统计

如果 solver 要冻结某些参数，compute 路径仍然必须产出完整的 `JtJ` 和 `Jty`，
否则活跃参数和冻结参数之间的交叉项会丢失。

实践中：

```python
ff_compute = Forcefield(trainer.forcefield)
ff_compute.param_mask = np.ones(ff_compute.n_params(), dtype=bool)
# 用 ff_compute 作为 run_post spec 里的 forcefield
```

### 规则 2：solver 内部应用 active mask

`FMMatrixSolver` 从其拥有的 `Forcefield.param_mask` 读取 active mask。

若 active index 为 `a`，frozen index 为 `f`，则在 shifted 系统上求解：

$$JtJ_{aa}\,\theta_a = Jty_a - JtJ_{af}\,\theta_f$$

这正确保留了非零冻结参数。

### 规则 3：solver mode 不依赖 optimizer 对象

Workflow 可以把 `trainer.optimizer.mask` 镜像到 `solver.forcefield.param_mask`，
但 solver 自身只依赖 `Forcefield.param_mask`。

---

## 求解路径

### OLS

在 active block 上进行对角缩放的最小二乘求解。

### Ridge

求解：

$$\left(JtJ_{aa} + \lambda I\right)\theta_a = Jty_a^*$$

其中 $Jty_a^*$ 是上方的 shifted 右端向量。

### Bayesian

在 active block 上进行对角 ARD 更新：

- 从 `weight_sum` 反推 raw 统计量
- 使用 Cholesky 分解和三角求解（不构建稠密矩阵逆）
- 在 `meta` 中报告收敛情况和后验超参数

---

## Workflow 最小调用模式

```python
import pickle

# 1. 读取 run_post 输出的 FM pickle
with open(spec_step["output_file"], "rb") as f:
    stats = pickle.load(f)

# 2. 构建 solver（使用全 mask forcefield 做 compute，active mask 做 solve）
ff_solve = copy.deepcopy(trainer.forcefield)
solver = FMMatrixSolver(ff_solve, mode="ridge", ridge_alpha=alpha)

# 3. 求解
result = solver.solve(stats)

# 4. 把求解结果传播回 trainer
trainer.update_forcefield(result["params"])
```
