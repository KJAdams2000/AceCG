# 09 Workflow 模块开发者参考

*Updated: 2026-04-23.*

> 本章只覆盖训练 / orchestration workflow。VP grower 单列在 [10_vp_grower.md](10_vp_grower.md)。

Workflow 层位于 scheduler / trainer / solver 之上，负责把配置、拓扑、资源和 batch 构造串起来。
它拥有 iteration 目录结构、checkpoint 语义、task 规划和 trainer / solver 调用时机；
它不拥有 reducer 数学、MPI broadcast、frame slicing 或 table/potential 内核。

---

## 核心文件

| 文件 | 职责 |
|---|---|
| `workflows/base.py` | `BaseWorkflow`、CLI override parser、topology / optimizer / resource builders |
| `workflows/sampling.py` | `SamplingWorkflow`、AA stats、forcefield staging、scheduler / sampler 构建 |
| `workflows/rem.py` | `REMWorkflow` |
| `workflows/cdrem.py` | `CDREMWorkflow` |
| `workflows/cdfm.py` | `CDFMWorkflow` |
| `workflows/fm.py` | `FMWorkflow` |
| `workflows/__init__.py` | 公开 workflow 类导出 |

---

## 层边界

```
config + topology + scheduler + trainer/solver
  → workflow 负责把这些对象接好

workflow
  → 规划任务、写 snapshot、读 post pickle、构建 batch、调用 trainer.step() / solver.solve()

compute / reducers / task_runner
  → 负责 task 内的一次 post 计算；workflow 只消费结果
```

Workflow 应该拥有：

- run 目录布局和 iteration 命名
- task spec 组装
- checkpoint / resume
- trainer / solver 选择和 batch schema 组装

Workflow 不应该拥有：

- 直接调用 reducer 作为主路径
- `MPIComputeEngine` 的内部调度细节
- trajectory 逐帧累积逻辑
- force / energy Jacobian 的数值细节

---

## `BaseWorkflow`

`BaseWorkflow` 是所有训练 workflow 的根基类。构造时只做三件事：

- 解析并应用 config override
- 建立 `output_dir`
- 由 `system.topology_file` 构建 `TopologyArrays`

最重要的共享 helper：

| 方法 | 作用 |
|---|---|
| `_run_workflow_cli()` | `acg-rem` / `acg-fm` 等 CLI 入口的统一封装 |
| `_apply_config_overrides()` | 支持 `--section.field value` 风格 override |
| `_build_topology()` | 基于当前 config 构建 `TopologyArrays` |
| `_build_optimizer()` | 从 `training.optimizer` / `training.trainer` 选择 optimizer |
| `_build_resource_pool()` | 从环境和 scheduler config 发现 MPI 后端和 CPU 资源 |
| `_build_forcefield_mask()` | 把 `[system] forcefield_mask` 编译成 runtime `param_mask` |

开发时最重要的约定：`BaseWorkflow.run()` 仍是抽象方法。真正的训练循环只能写在具体 workflow 里。

---

## `SamplingWorkflow`

`SamplingWorkflow` 是 REM / CDREM / CDFM 的共同基类，在 `BaseWorkflow` 之上新增：

- `ReadLmpFF()` 载入力场
- VP mask 和用户 `forcefield_mask` 的组合
- `TaskScheduler` 和 `BaseSampler` / `ConditionedSampler` 的装配
- `beta = 1 / (k_B T)` 的统一推导
- AA 参考数据策略和 workflow checkpoint I/O

关键 helper：

| 方法 | 作用 |
|---|---|
| `_build_forcefield()` | 从 LAMMPS settings 构建 runtime `Forcefield` |
| `_build_scheduler()` | 构建 `TaskScheduler` |
| `_build_sampler()` | 构建自由采样 `BaseSampler` |
| `_build_aa_data_strategy()` | REM 路径下选择常量 AA 统计或按 epoch 重算 |
| `_run_aa_post()` | 用当前 forcefield 在 AA 轨迹上跑一次 REM-style post |
| `_snapshot_forcefield()` | 为 MPI post 写 forcefield pickle |
| `_snapshot_optimizer()` | 保存 Adam/AdamW/RMSprop 的内部状态 |
| `_write_workflow_checkpoint()` | 保存完整 resume 状态 |
| `_load_workflow_checkpoint()` | 从上次完成的 epoch 恢复 |

`SamplingWorkflow` 本身不定义训练循环；它只是把“会跑模拟的 workflow”共通部分收口。

---

## 具体 workflow

### `REMWorkflow`

`REMWorkflow` 是最薄的单采样迭代器：

1. 每个 epoch 导出一次当前 forcefield
2. 通过 `BaseSampler` 启动一个 free-CG xz 任务
3. 读取 `step_mode="rem"` 的 `result.pkl`
4. 组装 `REMTrainerAnalytic.make_batch(...)`
5. `trainer.step(batch)` 后把参数写回 runtime forcefield

AA 侧统计由 `_build_aa_data_strategy()` 决定：线性路径可直接读缓存，非线性路径可按 epoch 重算。

### `CDREMWorkflow`

`CDREMWorkflow` 在 `SamplingWorkflow` 上再加一套 `ConditionedSampler`：

- xz 分支用自由采样 `BaseSampler`
- zbx 分支用 `ConditionedSampler`
- 每个 epoch 同时发出 `1 + K` 个任务

`_build_xz_task()` 和 `_build_zbx_task()` 都把 post 端统一到 `step_mode="cdrem"`，
下层 reducer 会复用 REM 路径。`_collect_cdrem_batch()` 负责把 xz 和 zbx 的
`result.pkl` 拼成 `CDREMBatch`。

### `CDFMWorkflow`

`CDFMWorkflow` 是 zbx-only workflow，没有 free xz 采样：

- `_build_sampler()` 被显式 override 成 `None`
- `init_config_pool` 和 `init_force_pool` 按 frame id 一一配对
- `_install_cdfm_mask()` 把默认训练 mask 收敛到 VP-only / mixed 通道
- 每个 zbx task 都走 `step_mode="cdfm_zbx"`

当前 active repo 里，`y_eff` 不再有单独的 `cdfm_y_eff` step。它在 `cdfm_zbx`
的 rank-0 预处理阶段从 `(init_config, init_force)` 一次性计算，然后广播给本 replica 的所有 rank。

### `FMWorkflow`

`FMWorkflow` 直接继承 `BaseWorkflow`，因为它不需要 sampler / scheduler 的迭代采样层。

它的特殊点有两个：

- `_build_fm_forcefield()` 从 `training.fm_specs` 构建训练用 forcefield
- `_run_post_accumulation()` 直接调用 `schedulers.task_runner.run_post(...)`
  在 AA 轨迹上累积 `step_mode="fm"` 统计

FM 有两条执行路径：

| 路径 | 条件 | 行为 |
|---|---|---|
| trainer path | `fm_method=iterator` 或 auto 选中一阶 optimizer | 每轮累积 FM 统计，再交给 `FMTrainerAnalytic.step()` |
| solver path | `fm_method=solver` 或 auto 选中 Newton/closed-form | 一次累积 FM 统计，再交给 `FMMatrixSolver.solve()` |

求解结束后，`_export_table_bundle()` 会把结果导出成 LAMMPS tables。

---

## 目录约定

Sampling workflows 的典型目录布局：

```
iter_0000/
  ff/
    forcefield_snapshot.pkl
    optimizer_snapshot.pkl
    workflow_checkpoint.pkl
  xz/      # REM / CDREM
  zbx/     # CDREM / CDFM
```

FM workflow 则使用：

```
fm_step_0000/
  forcefield.pkl
  fm_batch.pkl
```

对调试最重要的两个文件是：

- `ff/workflow_checkpoint.pkl`：resume 的规范入口
- 每个 replica 下的 `result.pkl`：compute post 的直接输出

---

## 开发规则

新增 workflow 时优先遵循这几个规则：

1. 只在 workflow 里决定 task 的数量、类型和 batch schema。
2. 只把 reducer 输出当成输入，不在 workflow 里复制 reducer 数学。
3. checkpoint 必须和 `Forcefield`、optimizer state、workflow RNG 一起保存。
4. VP grower 相关的一次性数据生产不要塞进训练 workflow，保持在 [10_vp_grower.md](10_vp_grower.md) 那条单独管线里。