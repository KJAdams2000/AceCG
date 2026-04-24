# 06 Optimizer 模块开发者参考

*Updated: 2026-04-23.*

Optimizer 是 trainer 使用的参数更新引擎。它们拥有更新状态和带 mask 的参数步进。
**不拥有** forcefield、compute runtime、帧统计或 workflow 控制。

---

## 核心模块

| 文件 | 职责 |
|---|---|
| `optimizers/base.py` | `BaseOptimizer`，共享 optimizer 合同 |
| `optimizers/adam.py` | 带 mask 的 Adam optimizer |
| `optimizers/adamW.py` | 带解耦 weight decay 的 AdamW optimizer |
| `optimizers/rmsprop.py` | 带 mask 的 RMSprop optimizer |
| `optimizers/newton_raphson.py` | 带 mask 的 Newton-Raphson optimizer |
| `optimizers/multithreaded/adam.py` | 特殊多线程 Adam 变体（非主线导出）|

`AceCG.optimizers.__init__` 目前只重新导出单进程主线 optimizer。

---

## 层边界

```
trainer
  → 计算 grad / hessian，决定是否调用 optimizer.step()

optimizer
  → mutate 自己的参数向量 L 并返回 delta_L

forcefield / workflow
  → 在 optimizer 外部；trainer 在 step 后同步它们
```

**Optimizer 应该拥有**：

- 当前扁平参数向量 `L`
- trainable mask `mask`
- 更新状态 buffer（如 moment、Hessian history）
- 更新规则

**Optimizer 不应该拥有**：

- `Forcefield`
- 参数 bounds
- MPI 或 reducer 状态
- 日志策略
- batch 构建

---

## 公开 Optimizer 接口

`AceCG.optimizers` 导出：

| 符号 | 含义 |
|---|---|
| `BaseOptimizer` | 抽象 optimizer 基类 |
| `AdamMaskedOptimizer` | 一阶 Adam |
| `AdamWMaskedOptimizer` | AdamW，解耦 weight decay |
| `RMSpropMaskedOptimizer` | RMSprop，支持 mask |
| `NewtonRaphsonOptimizer` | 二阶 Newton step |

---

## `BaseOptimizer`（optimizers/base.py）

### 状态

| 字段 | 含义 |
|---|---|
| `L` | 当前扁平参数向量 |
| `mask` | boolean trainable mask，shape 与 `L` 相同 |
| `lr` | 学习率 / 步长缩放参数 |

### 方法

| 方法 | 含义 |
|---|---|
| `__init__(L, mask, lr)` | 初始化参数向量、trainable mask 和学习率 |
| `set_params(L_new)` | 替换内部参数向量 |
| `state_dict()` | 序列化 optimizer 状态 |
| `load_state_dict(state)` | 恢复 optimizer 状态 |
| `step(grad, hessian=None)` | 抽象的单步更新 |

### 返回值约定

Optimizer 原地 mutate `self.L` 并返回全参数空间的变化量：

$$\Delta L = L_{\text{new}} - L_{\text{old}}$$

- masked-out 的坐标返回 0
- 梯度下降类 optimizer 通常返回负向量
- trainer 代码可以直接从返回值记录 `update_norm`

---

## Mask 语义

所有当前 optimizer 都是 mask-aware：

- 只有 `mask == True` 的坐标被更新
- masked-out 坐标保持不变
- 返回的 delta 向量始终是全长度

示例：

```python
opt = AdamMaskedOptimizer(L0, mask=np.array([True, False, True]), lr=1e-2)
delta = opt.step(grad)
# opt.L[1] 不变，delta[1] == 0
```

**重要**：`Forcefield.param_mask` / `key_mask` 是 trainability 的规范来源。
Optimizer 在初始化时接收 mask 副本作为执行状态；
workflow 不应把 `optimizer.mask` 当做独立的 source of truth。

---

## Hessian 合同

只有部分 optimizer 消耗 Hessian。Trainer 通过 `BaseTrainer.optimizer_accepts_hessian()` 发现，
后者检查 optimizer 的 `step()` 签名中是否有名为 `hessian` 的参数。

**开发者规则**：

- 若 optimizer 需要 Hessian，其 `step()` 签名**必须**包含 `hessian`
- 否则 trainer 代码将其视为一阶 optimizer

示例：

```python
def step(self, grad: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    ...
```

---

## 各 Optimizer 说明

### `AdamMaskedOptimizer`

内部状态：

- 一阶 moment `m`
- 二阶 moment `v`
- 步数计数器 `t`
- 可选预条件高斯噪声

更新规则：在 masked 坐标上执行标准 Adam；返回 `delta_L`。

### `AdamWMaskedOptimizer`

在 Adam 基础上增加：

- 解耦 weight decay
- 可选 AMSGrad 路径
- 可选预条件高斯噪声

当需要显式 weight decay 而非 L2-through-gradient 耦合时，使用这个。

### `RMSpropMaskedOptimizer`

内部状态：

- 运行中的梯度平方均值
- 可选 momentum buffer
- 可选 centered RMSprop 状态
- 可选噪声

支持：masked 更新、可选 L2 风格 weight decay（在梯度中）、可选 momentum。

### `NewtonRaphsonOptimizer`

同时消耗梯度和 Hessian，在 active block 上计算带 mask 的 Newton step。

是当前 Hessian-capable analytic trainer 使用的二阶路径。

额外存储用于下游日志：

- `last_grad`
- `last_hessian`
- `last_update`

### `optimizers/multithreaded/adam.py`

主线导出范围以外的特殊变体，视为实现细节扩展，不作为基准 optimizer 合同。

---

## 状态序列化

所有 optimizer 应支持：

```python
state = opt.state_dict()
opt.load_state_dict(state)
```

基础字段：

| Key | 含义 |
|---|---|
| `L` | 参数向量 |
| `mask` | trainable mask |
| `lr` | 学习率 |

子类在此基础上增加 moment buffer、计数器等内部状态。
