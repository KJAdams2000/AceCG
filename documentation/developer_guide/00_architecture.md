# 00 AceCG Software Architecture

*Updated: 2026-04-23. 合并并扩展自 draw.md（2026-04-03）。*

> 这是软件的总体架构文档，适合作为读代码的第一份参考。

---

## Executive Summary

AceCG 是一套 coarse-graining 力场训练系统，把 all-atom MD 轨迹作为输入，
通过多种训练方法（REM、FM、CDREM、CDFM 等）迭代优化 CG 力场参数。

核心设计哲学：**每层只做自己的事，向上只暴露 stable API，向下只依赖直接下层。**

---

## 层级总览

```
┌─────────────────────────────────────────────────────────────────┐
│  L6  Workflow                                                   │
│      workflows/base.py · sampling.py · rem.py · cdrem.py       │
│      fm.py · cdfm.py                                            │
│      vp_growth.py (standalone VP growth producer)               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  L5  Scheduler / Task Runner                                    │
│      schedulers/task_scheduler.py · task_runner.py             │
│      resource_pool.py · mpi_backend.py · profiler.py           │
└──────────────────────┬──────────────────────────────────────────┘
                       │ MPI task launch / pickle result
┌──────────────────────▼──────────────────────────────────────────┐
│  L4  Trainers / Solvers / Optimizers                            │
│      trainers/analytic/{rem,mse,fm,cdrem,cdfm,multi}.py        │
│      solvers/fm_matrix.py                                       │
│      optimizers/{adam,adamW,rmsprop,newton_raphson}.py          │
└──────────────────────┬──────────────────────────────────────────┘
                       │ batch dicts
┌──────────────────────▼──────────────────────────────────────────┐
│  L3  Compute Runtime                                            │
│      compute/mpi_engine.py  (MPIComputeEngine, run_post)       │
│      compute/frame_geometry.py · energy.py · force.py          │
│      compute/reducers.py   (stateful pipeline reducers)        │
│      compute/registry.py   (build_default_engine)              │
│      analysis/rdf.py       (RDF/PDF post-processing)           │
└──────────────────────┬──────────────────────────────────────────┘
                       │ TopologyArrays + Forcefield + frames
┌──────────────────────▼──────────────────────────────────────────┐
│  L2  I/O + Config                                               │
│      io/trajectory.py · forcefield.py · coordinates.py        │
│      io/tables.py · logger.py                                   │
│      configs/parser.py · models.py · vp_config.py             │
└──────────────────────┬──────────────────────────────────────────┘
                       │ TopologyArrays, Forcefield
┌──────────────────────▼──────────────────────────────────────────┐
│  L1  Topology / Forcefield                                      │
│      topology/types.py      (InteractionKey)                   │
│      topology/topology_array.py  (TopologyArrays)             │
│      topology/forcefield.py (Forcefield)                       │
│      topology/neighbor.py   (pair routing / exclusions)        │
│      topology/mscg.py       (MS-CG topology helper)           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  L0  Potentials / Fitters                                       │
│      potentials/{harmonic,bspline,gaussian,lj,...}.py          │
│      fitters/{fit_bspline,fit_harmonic,fit_multi_gaussian}.py  │
│      samplers/base.py · conditioned.py                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 各层详细说明

### L0：Potentials / Fitters / Samplers

**不依赖任何其他 AceCG 模块**，是系统的叶子层。

| 模块 | 职责 |
|---|---|
| `potentials/base.py` | `BasePotential` 抽象基类；`IteratePotentials` 遍历帮手 |
| `potentials/harmonic.py` | 谐振 bonded / angle potential |
| `potentials/bspline.py` | force-basis B-spline potential（所有参数线性） |
| `potentials/gaussian.py` | 归一化 Gaussian pair potential |
| `potentials/lj*.py` | LJ 12-6 / 9-6 / soft-core |
| `potentials/multi_gaussian.py` | 归一化 multi-Gaussian family |
| `fitters/fit_*.py` | 从 RDF/distribution 拟合初始参数 |
| `samplers/base.py` | LAMMPS 脚本构建基类 |
| `samplers/conditioned.py` | conditioned sampling 脚本构建 |

**这层不拥有**：atom indexing、MPI、frame iteration、trainer 状态。

标量力学约定：`F(r) = -dU/dr`，compute 层负责把标量 F 投影回
Cartesian atom forces（通过 `FrameGeometry` 里的几何方向向量）。

---

### L1：Topology / Forcefield

核心三件套：

| 类 | 文件 | 职责 |
|---|---|---|
| `InteractionKey` | `topology/types.py` | 规范化的相互作用标识符 `NamedTuple(style, types)` |
| `TopologyArrays` | `topology/topology_array.py` | 冻结的 MPI-broadcastable 拓扑快照 |
| `Forcefield` | `topology/forcefield.py` | `MutableMapping[InteractionKey, List[BasePotential]]` + 参数向量缓存 |

**`InteractionKey` 规范化规则**：

| style | 规则 | 例子 |
|---|---|---|
| pair / bond | 字母序：`(a,b) if a ≤ b` | `pair("C","A")` → `("A","C")` |
| angle | 反转如果 `a > c`：`(a,b,c) if a ≤ c else (c,b,a)` | `angle("Z","Y","A")` → `("A","Y","Z")` |
| dihedral | 反转如果 `(a,b) > (d,c)` | `dihedral("D","C","B","A")` → `("A","B","C","D")` |

**`TopologyArrays`** 由 `collect_topology_arrays(universe, ...)` 一次性构建，
然后广播给所有 MPI rank。包含：
- 原子级数组（names, types, masses, charges, atom_type_codes）
- 成键项（bonds, angles, dihedrals）
- 排除列表（exclude_12, exclude_13, exclude_14）
- 虚位点分类（real_site_indices, virtual_site_mask）
- 实例→类型映射（bond_key_index, keys_bondtypes 等）

**`Forcefield`** 是整个参数系统的 source of truth：
- `param_array()` / `update_params(L)` 控制整个参数向量
- `key_mask` / `param_mask` 控制哪些参数参与训练
- `deepcopy()` / `copy.deepcopy()` 均安全（trainer、solver 各自拥有独立副本）

---

### L2：I/O + Config

| 模块 | 职责 |
|---|---|
| `io/trajectory.py` | `iter_frames()` — unified trajectory reader，产出 `(frame_id, positions, box, forces)` |
| `io/forcefield.py` | `ReadLmpFF` / `WriteLmpFF` — LAMMPS force field 读写 |
| `io/coordinates.py` | CG coordinate mapping helpers |
| `io/coordinates_writers.py` | LAMMPS data 文件写入（含 topology 支持）|
| `io/tables.py` | LAMMPS tabulated potential 读写 |
| `io/logger.py` | 结构化日志 |
| `configs/parser.py` | `.acg` 配置文件解析 |
| `configs/models.py` | 配置数据类（`SchedulerConfig`、`WorkflowConfig` 等） |
| `configs/vp_config.py` | VP 相关配置 |

重要运行时约定：

- `iter_frames()` 是 compute engine 的统一 trajectory 入口
- 拓扑由 task spec 显式提供，每次 `run_post()` 重建一次 `TopologyArrays`
- 力场快照以 pickle 文件传递（`forcefield_path` 字段）

详见 [11_io_utilities.md](11_io_utilities.md)。

---

### L3：Compute Runtime

这是系统数值计算的核心。详见 [03_compute.md](03_compute.md) 和 [04_mpi_runtime.md](04_mpi_runtime.md)。

**核心模块**：

| 文件 | 职责 |
|---|---|
| `compute/mpi_engine.py` | `MPIComputeEngine`、`FrameCache`、`TrajectoryCache`（legacy aliases: `FrameObservables`、`TrajectoryObservablesCache`） |
| `compute/frame_geometry.py` | `FrameGeometry` — 不可变的 per-frame 几何视图 |
| `compute/energy.py` | `energy()` 内核 |
| `compute/force.py` | `force()` 内核 |
| `compute/reducers.py` | 有状态 pipeline reducer（init/consume/finalize）|
| `compute/registry.py` | `build_default_engine()` — 注册核心 observable |
| `analysis/rdf.py` | RDF / PDF 分布函数计算 |

**三个持久 public API**：

```python
FrameGeometry = compute_frame_geometry(positions, box, topology_arrays, ...)
energy_dict   = energy(geom, forcefield, return_grad=True, ...)
force_dict    = force(geom, forcefield, return_grad=True, ...)
```

**这层不拥有**：scheduler 策略、workflow orchestration、跨 task 的 trainer tally。

---

### L4：Trainers / Solvers / Optimizers

| 模块 | 职责 |
|---|---|
| `trainers/base.py` | `BaseTrainer` 抽象约定 |
| `trainers/analytic/rem.py` | `REMTrainerAnalytic` — 能量梯度批次消耗 |
| `trainers/analytic/fm.py` | `FMTrainerAnalytic` — 迭代 FM 批次消耗 |
| `trainers/analytic/cdrem.py` | `CDREMTrainerAnalytic` — 潜变量 REM |
| `trainers/analytic/cdfm.py` | `CDFMTrainerAnalytic` — 按 x 聚合的 CDFM |
| `trainers/analytic/mse.py` | `MSETrainerAnalytic` — PMF 匹配 MSE |
| `trainers/analytic/multi.py` | `MultiTrainerAnalytic` — 多子训练器组合 |
| `solvers/fm_matrix.py` | `FMMatrixSolver` — 精确 FM 矩阵求解（OLS/ridge/Bayesian） |
| `optimizers/adam.py` | 带 mask 的 Adam |
| `optimizers/newton_raphson.py` | 二阶 Newton-Raphson |

**所有 trainer 的 step 合同**：

```python
out = trainer.step(batch, apply_update=True)
```

batch 由 workflow 构建，trainer 不重建 trajectory 状态。

---

### L5：Scheduler

详见 [08_schedulers.md](08_schedulers.md)。

三层设计：资源发现 → 贪心分配 → MPI 命令组装。

Workflow 只需：`TaskScheduler.run_iteration(xz_tasks, zbx_tasks)` → 返回 `list[TaskResult]`。

---

### L6：Workflow

Workflow 是 orchestration 层，应尽量薄。

| 文件 | 职责 |
|---|---|
| `workflows/base.py` | `BaseWorkflow`、CLI override helpers、topology / optimizer / resource builders |
| `workflows/sampling.py` | `SamplingWorkflow`、AA 统计、forcefield staging、sampler / scheduler 构建 |
| `workflows/rem.py` | REM workflow |
| `workflows/cdrem.py` | CDREM workflow（xz 阶段 + zbx 阶段） |
| `workflows/fm.py` | FM workflow（iterative trainer 或闭式 solver） |
| `workflows/cdfm.py` | CDFM workflow（zbx only；`cdfm_zbx` 内含 `y_eff` 预处理）|
| `workflows/vp_growth.py` | standalone VP growth workflow（一次性数据生产） |

详见 [09_workflows.md](09_workflows.md) 和 [10_vp_grower.md](10_vp_grower.md)。

**合法 workflow 应该做的**：

- 决定启动哪些 task
- 调用 `trainer.step(batch)` 或 `solver.solve(batch)`
- 读取 compute task 输出的 pickle
- 组装合法的 batch dict
- 管理 iteration 计数和 checkpoint

**合法 workflow 不应该做的**：

- 直接调用 reducers 作为主要生产路径
- 自己拥有 MPI broadcast、frame slicing、cache lifetime
- 复现 compute 侧的聚合逻辑（哪怕 legacy helper 存在）

---

## 完整数据流图

```
AA 轨迹 (.lammpstrj)
         │
         ▼
  io/trajectory.py
  iter_frames(trajectory, topology)
         │
         │ (frame_id, positions, box, forces)
         ▼
  compute/mpi_engine.py
  MPIComputeEngine.run_post(spec)
    ├── rank 0: 加载 forcefield、Universe、TopologyArrays、sel_indices
    ├── MPI broadcast: Universe, TopologyArrays, sel_indices
    ├── 每个 rank 分到一段连续帧
    ├── iter_frames → 本地帧序列
    └── for each step in spec["steps"]:
           │
           ├── step_mode="rem"  → reducers.init_rem_state / consume_rem_frame
           ├── step_mode="fm"   → reducers.init_fm_state  / consume_fm_frame
           ├── step_mode="cdrem"→ 同 rem
           ├── step_mode="cdfm_zbx"   → reducers.init_cdfm_zbx_state / ...
           │                             （rank 0 先算一次 y_eff）
           └── step_mode="rdf"        → analysis/rdf.py (不进 one-pass pipeline)
                    │
                    │ local partials
                    ▼
         comm.reduce / comm.gather (rank 0)
                    │
                    ▼
         finalize_*_root → pickle output
                    │
         ┌──────────▼──────────────┐
         │   trainer batch dict    │
         └──────────┬──────────────┘
                    │
                    ▼
         trainer.step(batch)
           ├── compute gradient ∇L
           ├── (optionally) compute Hessian H
           └── optimizer.step(grad, hessian) → ΔL
                    │
                    ▼
         forcefield.update_params(L + ΔL)
                    │
                    ▼
         checkpoint 保存
         下一 iteration
```

---

## 核心类依赖图

```
InteractionKey
    ├── TopologyArrays (持有 bond_key_index, keys_bondtypes 等)
    └── Forcefield (key → List[BasePotential])

BasePotential
    └── Forcefield

TopologyArrays + Forcefield + positions/box
    └── FrameGeometry (compute_frame_geometry)
            ├── energy(geom, ff) → {energy, energy_grad, ...}
            └── force(geom, ff)  → {force, force_grad, fm_stats}

MPIComputeEngine
    ├── 拥有: MPI comm, _registry
    ├── compute(request, frame, ...) → local observable dict
    └── run_post(spec) → pickle files
            └── 使用: reducers.py pipeline (init/consume/finalize)

reducers.py
    ├── 输入: engine.compute() 结果 + topology + forcefield
    └── 输出: 本地 partial dicts（供 MPI reduce 使用）

trainer.step(batch)
    ├── 输入: workflow 组装的 batch dict（来自 pickle）
    ├── 计算: ∇L, H（可选）
    └── 输出: optimizer.step(grad) → ΔL → forcefield 更新

workflow / scheduler
    └── orchestrate 以上，但不吸收其内部逻辑
```

---

## 实际代码阅读顺序

如果需要快速理解当前 runtime，按以下顺序读：

1. `topology/types.py` — `InteractionKey`
2. `topology/topology_array.py` — `TopologyArrays`、`collect_topology_arrays()`
3. `topology/forcefield.py` — `Forcefield`、参数向量 API
4. `compute/frame_geometry.py` — `FrameGeometry`、`compute_frame_geometry()`
5. `compute/registry.py` — `build_default_engine()`、注册了哪些 observable
6. `compute/mpi_engine.py` — `MPIComputeEngine.compute()`、`run_post()`
7. `compute/reducers.py` — init/consume/finalize pipeline API
8. `trainers/analytic/*.py` — 各 trainer 的 batch 合同和 step() 实现
9. `solvers/fm_matrix.py` — 矩阵 FM 求解
10. `schedulers/task_runner.py` — 计算节点上发生了什么
11. `workflows/base.py` — checkpoint、iteration loop 基础设施

---

## 重要设计边界

| 关系 | 允许 | 禁止 |
|---|---|---|
| workflow → compute | 调用 `MPIComputeEngine.run_post()`，消耗 pickle | 直接调用 reducer，自己做 MPI reduce |
| workflow → trainer | 构建 batch，调用 `trainer.step()` | 复现 trainer 内部的梯度计算 |
| trainer → compute | 无（trainer 不创建 compute engine）| 调用 `run_post()`，创建 `MPIComputeEngine` |
| scheduler → compute | 启动 `mpi_engine` 子进程（`task_runner`）| 直接实例化 `MPIComputeEngine` |
| reducer → compute | 调用 `engine.compute()` 本地 | 读轨迹，MPI broadcast，写输出文件 |
| 所有层 → topology | import `InteractionKey`、`TopologyArrays`、`Forcefield` | 不可反向依赖 |
