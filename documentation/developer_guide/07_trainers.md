# 07 Trainer 模块开发者参考

*Updated: 2026-04-23.*

Trainer 层位于 `compute/` 之上，与 `solvers/` 同层。
它消耗 workflow 预构建的 batch 统计量，产出梯度、Hessian、更新步和诊断信息。
**不拥有** MPI 执行、trajectory 提取、或 reducer runtime 状态。

---

## 核心模块

| 文件 | 职责 |
|---|---|
| `trainers/base.py` | `BaseTrainer`，共享 trainer 合同 |
| `trainers/analytic/rem.py` | REM trainer + batch schema |
| `trainers/analytic/mse.py` | PMF-matching MSE trainer（基于 REM 能量统计）|
| `trainers/analytic/fm.py` | 迭代 FM trainer，消耗标准 FM reducer payload |
| `trainers/analytic/cdrem.py` | 潜变量 CDREM trainer |
| `trainers/analytic/cdfm.py` | CDFM 梯度消耗器，含 EM guardrail 处理 |
| `trainers/analytic/multi.py` | 组合多个子 trainer 的 meta-trainer |
| `trainers/autodiff/` | 未来 autodiff trainer 的占位 package |

`AceCG.trainers` 目前只重新导出 analytic trainers。

---

## 层边界

```
topology + compute
  → Forcefield-owned masks、reducer 输出、workflow 构建的 batch

trainer
  → 消耗一个 batch，计算 grad/update，mutate 自己的 Forcefield + optimizer

workflow
  → 决定构建哪个 batch、何时调用 trainer.step()、如何记录日志
```

**Trainer 应该拥有**：

- 一个私有 `Forcefield` 副本
- 一个私有 optimizer 实例
- batch → 梯度 的数学
- trainer-local 日志和诊断

**Trainer 不应该拥有**：

- `MPIComputeEngine`
- `run_post()`
- trajectory I/O 或 MDAnalysis 对象
- reducer 侧 accumulation 循环
- solver 式精确线性求解

---

## `BaseTrainer`（trainers/base.py）

### 职责

| 方法 | 含义 |
|---|---|
| `__init__(forcefield, optimizer, beta=None, logger=None)` | deep-copy 并拥有一个 `Forcefield` 和 `BaseOptimizer` |
| `get_params()` | 返回当前全参数向量 |
| `update_forcefield(L_new)` | 将全参数向量写入 forcefield 和 optimizer |
| `clamp_and_update()` | 将 bounds 应用到 optimizer 状态，再同步 |
| `get_param_names()` | 返回有序参数标签 |
| `get_param_bounds()` | 返回 `(lb, ub)` |
| `get_interaction_labels()` | 返回 forcefield 顺序的 interaction 标签 |
| `n_total_params()` | 总标量参数数 |
| `active_interaction_mask()` | 从 optimizer L2 mask 推导 L1 interaction 活跃性 |
| `is_optimization_linear()` | 报告是否所有活跃通道都是线性的 |
| `optimizer_accepts_hessian()` | 基于签名判断 optimizer 是否接受 Hessian |
| `step(batch, apply_update=True)` | 抽象的单步 trainer 入口 |

### Batch 合同

每个 concrete trainer 都是 batch-driven：

```python
out = trainer.step(batch, apply_update=True)
```

Workflow 拥有 batch 构建。**Trainer 不应重建 trajectory 状态或自行补算缺失的帧统计。**

### Hessian 合同

`BaseTrainer.optimizer_accepts_hessian()` 检查 optimizer 的 `step()` 签名中是否有名为 `hessian` 的参数。
因此 optimizer 作者若要支持二阶信息，**必须**用精确名称 `hessian`。

---

## 公开 Trainer 接口

`AceCG.trainers` 导出：

| 符号 | 含义 |
|---|---|
| `REMTrainerAnalytic`, `REMBatch`, `REMOut` | REM trainer |
| `MSETrainerAnalytic`, `MSEBatch`, `MSEOut` | MSE / PMF-matching trainer |
| `FMTrainerAnalytic`, `FMBatch` | 迭代 FM trainer |
| `CDREMTrainerAnalytic`, `CDREMBatch`, `CDREMOut` | 潜变量 REM trainer |
| `CDFMTrainerAnalytic`, `CDFMBatch` | 潜变量 FM trainer |
| `MultiTrainerAnalytic`, `MultiOut` | 组合 meta-trainer |

推荐用法：

- workflow 调用 `TrainerClass.make_batch(...)` 或等效 batch helper
- workflow 把 batch 传入 `trainer.step(...)`
- trainer 内部不创建 compute engine、不调用 `run_post()`、不跑 reducer

---

## Analytic Trainers

### `REMTrainerAnalytic`

消耗 REM 能量统计，计算：

$$\nabla = \beta \left(\langle dU / d\lambda \rangle_{\text{AA}} - \langle dU / d\lambda \rangle_{\text{CG}}\right)$$

AA 侧统计来自 workflow 构建的 batch（`run_post(step_mode="rem")`的 pickle 输出）。
当 optimizer 接受 Hessian 时，REM 也消耗二阶统计。

### `FMTrainerAnalytic`

消耗标准 FM reducer payload：

| Key | 含义 |
|---|---|
| `JtJ` | 归一化加权平均 $J_i^T J_i$ |
| `Jty` | 归一化加权平均 $J_i^T y_i$ |
| `y_sumsq` | 归一化加权平均 $y_i^T y_i$ |
| `Jtf` | 归一化加权平均 $J_i^T f_i$ |
| `f_sumsq` | 归一化加权平均 $f_i^T f_i$ |
| `fty` | 归一化加权平均 $f_i^T y_i$ |
| `nframe` | 贡献帧数 |

计算：

$$L = \frac{1}{2}\left(f\_\text{sumsq} - 2\,fty + y\_\text{sumsq}\right)$$
$$\nabla = Jtf - Jty,\quad H \approx JtJ$$

这是**迭代 FM 路径**。精确闭合求解在 `FMMatrixSolver`，不在 trainer。

FM stats 由 `run_post(spec, step_mode="fm")` 生成（pickle 输出），workflow 读取后构建 batch。

### `MSETrainerAnalytic`

消耗 PMF + REM 能量梯度统计，构建 gauge-fixed PMF 误配梯度。
使用：

$$\partial F_{\text{CG}}(s)/\partial \lambda = \langle dU/d\lambda \rangle_{\text{CG}|s} - \langle dU/d\lambda \rangle_{\text{CG}}$$

然后对各 bin 累积 PMF 误配梯度。

**已知问题**：当 workflow 提供非均匀帧权重给 REM reducer 时，
全局均值和条件均值会不一致（未来需修复：在 `MSEBatch` 加入 `frame_weight`）。

### `CDREMTrainerAnalytic`

消耗潜变量条件和联合导数统计，计算 CDREM 梯度和可选 Hessian。

### `CDFMTrainerAnalytic`

消耗 by-`x` batch 数组：

- `grad_direct_by_x`
- `grad_reinforce_by_x`
- `sse_by_x`
- `n_samples_by_x`
- `obs_rows`
- 可选 `x_weight`
- `mode`

在 `step()` 内部对各 `x` 做 tally，然后应用 guardrail、REINFORCE clipping、mask 和 optimizer step。
**不自行构建 trajectory 统计。**

#### CDFM 数据通道

CDFM 不再使用 `cdfm_y_eff` 预处理 step。每个 zbx replica 在 `.acg` 里通过
两个配对的 glob 模式声明：

```ini
[conditioning]
init_config_pool = conditioning/frame_*.data
init_force_pool  = conditioning/frame_*.forces.npy
# 默认 True：CDFM 梯度更新只驱动 VP 项，CG-only 参数被冻结
mask_cg_only     = true
```

两个 pool 的帧 id 必须一一匹配；`init_force_pool` 的文件名通过
`AceCG.configs.utils.extract_frame_id_from_force_file()` 提取数字 id，
多个数字 run 必须完全一致（如 `frame_000035.forces.npy` 或 `frame_35_rep35.npy`）。

`CDFMWorkflow` 在 `__init__` 阶段根据 `mask_cg_only` 调用
`forcefield.build_mask(init_mask=init_mask)`：当 `True` 时将当前 mask 与
`~real_mask` 做 AND 合成，把所有 CG-only（非 VP）项冻结；`False` 时保留
现有 mask，允许 CG-only 通道参与 CDFM 梯度更新。对 CG-only 力基线 \
`f_theta_cg_only(R_init)` 的评估由 `run_post()` 的 cdfm_zbx 预处理块\
独立处理（临时 swap 到 `real_mask`，计算完成后恢复），不受 `mask_cg_only` 影响。

### `MultiTrainerAnalytic`

在 trainer 层组合多个子 trainer。两种模式：

| Mode | 含义 |
|---|---|
| `update` | 每个子 trainer 独立 update；meta-trainer 合并更新 |
| `grad` | 每个子 trainer dry-run 返回 grad/Hessian；meta-trainer 统一 optimizer step |

`MultiTrainerAnalytic` 不是 compute 共享对象。共享 reducer 工作和 cache 复用必须在下层（workflow / compute）解决。

---

## Workflow 合同

Workflow 负责：

1. 选择使用哪个 trainer
2. 构建正确的 batch dict
3. 调用 `trainer.step(batch, apply_update=...)`
4. 决定是否切换到 solver 模式
5. 日志和 checkpoint 策略

最小调用模式（FM 示例）：

```python
batch = FMTrainerAnalytic.make_batch(
    JtJ=stats["JtJ"],
    Jty=stats["Jty"],
    y_sumsq=stats["y_sumsq"],
    Jtf=stats["Jtf"],
    f_sumsq=stats["f_sumsq"],
    fty=stats["fty"],
    nframe=stats["nframe"],
    step_index=iteration_index,
)
out = trainer.step(batch, apply_update=True)
```

这里 `stats` 是从 `run_post(spec)` 输出的 pickle 文件中读取的。
