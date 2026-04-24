# 04 AceCG MPI Runtime 与任务调用链

*Updated: 2026-04-23. 合并并扩展自 draw_mpi.md（2026-04-03）。*

---

## 概述

这份文档描述：

- 一个 scheduled task 如何从 workflow 到达 compute runtime
- `MPIComputeEngine.run_post()` 实际做了什么
- `step_mode="multi"` 的语义
- MPI 逻辑住在哪里（以及哪里不住）
- reducer pipeline（init / consume / finalize）的设计

---

## 端到端调用链

```
workflow
    │
    ▼  构建 task spec (dict)
scheduler builds task spec
    │
    ▼
task_runner.py 启动（subprocess）
    │
    ├── Phase 1: subprocess.Popen(sim_launch.argv)  ← LAMMPS 运行
    │
    └── Phase 2: MPI post（如果 spec 里有 steps）
              │
              ├── in-process:  build_default_engine().run_post(spec)
              └── MPI:         python -m AceCG.compute.mpi_engine spec.json
                                       │
                                       ▼
                               mpi_engine.__main__
                               build_default_engine(comm=comm)
                               engine.run_post(spec)
```

---

## Stage 1：Workflow / Scheduler 边界

Workflow 和 scheduler 只做三件事：

1. 构建合法的 task spec
2. 启动 task
3. 消费 pickle 输出

它们不应该拥有：

- trajectory broadcast
- MPI reduction
- frame chunking
- reducer internals

---

## Stage 2：`task_runner.py`

`task_runner.py` 是计算节点上的本地包装器：

```
task_runner.run(spec):
  ┌─ 模拟阶段 ───────────────────────────────────────────────┐
  │  subprocess.Popen(spec["sim_launch"]["argv"], env=...)   │
  │  wait for exit code                                      │
  └──────────────────────────────────────────────────────────┘
  ┌─ 后处理阶段（可选）───────────────────────────────────────┐
  │  如果 spec 有 "steps":                                   │
  │    in-process:  build_default_engine().run_post(spec)    │
  │    MPI post:    subprocess.Popen(post_launch.argv, ...)  │
  └──────────────────────────────────────────────────────────┘
```

---

## Stage 3：`mpi_engine.__main__`

入口行为很薄：

```python
spec = json.load(sys.argv[1])
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None
engine = build_default_engine(comm=comm)
engine.run_post(spec)
```

没有 module-level `run_post()` shim。

---

## Stage 4：`build_default_engine()`

`build_default_engine()` 创建一个 `MPIComputeEngine`，注册 7 个核心 observable：

| 名称 | reduce 模式 | 描述 |
|---|---|---|
| `energy_grad` | gather | per-frame 能量梯度 dU/dθ |
| `energy_value` | gather | per-frame 标量能量 |
| `energy_hessian` | gather | per-frame 能量 Hessian d²U/dθ² |
| `energy_grad_outer` | gather | per-frame 梯度外积 |
| `force_grad` | gather | per-frame 力 Jacobian df/dθ |
| `force_value` | gather | per-frame model forces |
| `fm_stats` | dict_sum | per-frame FM sufficient statistics（JtJ, Jty, y_sumsq） |

---

## Stage 5：`MPIComputeEngine.run_post(spec)`

这是真正的 MPI 后处理边界，逻辑集中在 `compute/mpi_engine.py`。

### spec 结构

```python
spec = {
    "work_dir":        str,
    "forcefield_path": str,          # pickle 路径
    "topology":        str,          # LAMMPS data 或 PSF 路径
    "trajectory":      str | list,   # lammpstrj 路径
    "trajectory_format": str,        # 默认 "LAMMPSDUMP"
    "frame_start":     int,          # 可选
    "frame_end":       int,          # 可选
    "every":           int,          # 可选，默认 1
    "frame_ids":       list[int],    # 可选，离散帧选择
    "frame_weight":    list[float],  # 可选
    "cutoff":          float,        # 可选，pair 搜索截断
    "sel":             str,          # MDAnalysis 选择表达式，默认 "all"
    "exclude_bonded":  str,          # "111" 等，默认 "111"
    "exclude_option":  str,          # "resid"/"none" 等
    "atom_type_name_aliases": dict,  # 可选，LAMMPS type int → name
    "vp_names":        list[str],    # 可选，虚位点名称
    "collect_observables": bool,     # 是否收集 FrameCache
    "gather_observables":  bool,     # 是否 gather 到 rank 0
    "steps": [                       # 一个或多个 step
        {
            "step_mode":   str,      # 见下表
            "output_file": str,      # pickle 输出路径
            ...                      # step-specific keys
        },
        ...
    ]
}
```

### step_mode 汇总

| `step_mode` | 用途 | 主要输出 key |
|---|---|---|
| `rem` | REM / AA-side 能量统计 | `energy_grad_avg`, `n_frames`, `weight_sum`, `d2U_avg`（可选）|
| `cdrem` | CDREM xz 或 zbx 后处理（与 rem 同一 reducer）| 同 rem |
| `fm` | Force matching 统计 | `JtJ`, `Jty`, `y_sumsq`, `Jtf`, `f_sumsq`, `fty`, `nframe` |
| `cdfm_zbx` | CDFM conditioned sampling replica；rank 0 先从 `(init_config, init_force)` 计算 `y_eff` | `grad_direct`, `grad_reinforce`, `sse`, `obs_rows`, `n_samples` |
| `rdf` | RDF / PDF 分布函数（不进 one-pass pipeline）| `{InteractionKey: distribution_array}` |

> `cdfm_y_eff` 已从 active repo 移除。`y_eff` 预处理现在并入 `cdfm_zbx` 的 rank-0 前置逻辑。

### 高层流程图

```
run_post(spec):
  ┌─ rank 0 only ──────────────────────────────────────────┐
  │  load forcefield pickle                                │
  │  mda.Universe(topology, trajectory)                    │
  │  collect_topology_arrays(universe, ...)                │
  │  sel_indices = universe.select_atoms(sel).indices      │
  │  total_frames = len(universe.trajectory)               │
  └────────────────────────────────────────────────────────┘
                     │
                     │ comm.bcast((universe, topology_arrays,
                     │             sel_indices, total_frames))
                     ▼
  ┌─ all ranks ────────────────────────────────────────────┐
  │  帧分配（连续 chunk per rank，或 discrete_ids 均分）   │
  │  one_pass_steps = [steps where step_mode != "rdf"]     │
  │  rdf_steps      = [steps where step_mode == "rdf"]     │
  └────────────────────────────────────────────────────────┘
                     │
       ┌─────────────▼───────────────┐
       │  one-pass pipeline          │
       │  for each frame in local:   │
       │    engine.compute(request)  │
       │    for step in one_pass_steps:
       │      consume_*_frame(state, frame_result)
       └─────────────┬───────────────┘
                     │ local partials
       ┌─────────────▼───────────────┐
       │  MPI reduce (per step)      │
       │  comm.reduce(local_partials,│
       │              root=0)        │
       └─────────────┬───────────────┘
                     │ (rank 0 only)
       ┌─────────────▼───────────────┐
       │  finalize_*_root(state)     │
       │  pickle.dump(result,        │
       │    open(output_file, "wb")) │
       └────────────────────────────-┘
                     │
       (rdf_steps 单独走 analysis.rdf，
        不进 one-pass pipeline)
```

---

## 帧分配策略

两种模式：

**连续 chunk（默认）**：

```
[frame_start, frame_end, every] → n_selected frames
rank i 拿到: [local_offset, local_offset + local_count) * every
```

特殊情况：如果 `n_selected ≤ MPI size`，分布式 slicing 禁用，rank 0 处理全部帧。

**离散帧选择（`frame_ids`）**：

- opt-in via `spec["frame_ids"]`，用于任何需要显式 frame-id 子集的 post task
- 均分到各 rank

---

## Reducer Pipeline API

`reducers.py` 为每种 step_mode 提供五个函数，构成有状态 pipeline：

```
init_*_state(step) → state dict
request_*(step)    → {need_energy_grad: bool, need_force_grad: bool, ...}
consume_*_frame(state, frame_result, ...)  → mutates state
local_partials_*(state)  → partial dict（供 comm.reduce 使用）
reduce_plan_*(step)  → {key: ("sum"|"gather"|...) for key in partials}
finalize_*_root(state) → final result dict
```

当前有 4 套：

| prefix | step_mode |
|---|---|
| `init_fm_state` / ... | `fm` |
| `init_rem_state` / ... | `rem`, `cdrem` |
| `init_cdfm_zbx_state` / ... | `cdfm_zbx` |
| （直接走 `analysis.rdf`）| `rdf` |

`cdfm_zbx` 的 reducer 在主 one-pass loop 前先完成一次 `y_eff` 预处理，然后再进入逐帧累计。

这套设计的好处：

- 每个 step 的 local accumulation 在独立 state 里，互不干扰
- MPI reduce 的具体操作（sum/gather）由 `reduce_plan_*` 声明，engine 统一执行
- reducer 函数是纯 local 数学，**不做任何 MPI 或 I/O**

---

## Observable Cache

`MPIComputeEngine.run_post()` 支持可选的 observable 缓存，
用于 workflow 需要 per-frame 几何信息（如 RDF 分析、可视化）时：

```python
spec["collect_observables"] = True   # 每帧构建 FrameCache
spec["gather_observables"]  = True   # MPI: gather 到 rank 0
spec["observables_output_file"] = "traj_cache.pkl"  # 可选，写 pickle
```

相关类：

| 类 | 位置 | 职责 |
|---|---|---|
| `FrameCache` | `compute/mpi_engine.py` | 轻量级 per-frame 几何 digest（pair distances、bonds、angles 等） |
| `TrajectoryCache` | `compute/mpi_engine.py` | 多帧 `FrameCache` 的集合体 |
| `geometry_to_observables()` | `compute/mpi_engine.py` | 从 `FrameGeometry` 切出 `FrameCache` |

旧名字 `FrameObservables` / `TrajectoryObservablesCache` 仍保留为 compatibility
aliases，但新的文档和新代码应优先使用 `FrameCache` /
`TrajectoryCache`。

---

## MPI 逻辑归属

### 在 `MPIComputeEngine.run_post()` 里

- rank 0 加载共享上下文（forcefield, universe, topology, sel_indices）
- 共享上下文 broadcast
- 连续帧切分
- 本地帧提取

### 在 reducer loop 里（仍属于 `run_post()`）

- 对每个 one-pass step：`comm.reduce(local_partials, root=0)` 或 `comm.gather()`
- rank 0：`finalize_*_root()` + pickle 写出

### 不在 reducers 里

- 无轨迹读取
- 无 MPI broadcast
- 无输出文件写入

### 不在 workflows 里

- 无 MPI frame slicing
- 无 engine cache 所有权
- 无 reducer 侧 MPI 数学

---

## 合法的下游访问模式

```
workflow / scheduler
    │
    ▼  构建 task spec（含 steps）
MPIComputeEngine.run_post(spec)
    │
    ▼  pickle result files
trainer.step(batch) 或 solver.solve(batch)
    │
    ▼  Forcefield 更新
```

**合法**：

- 启动 MPI post tasks
- 读 pickle 输出
- 组装 trainer / solver batch

**非法 / 强烈不建议**：

- 直接调 reducers 作为 workflow 的主要生产接口
- workflow 代码里自己做 MPI frame splitting
- workflow 代码里拥有 cache invalidation
- 因 legacy helper 函数存在就在 workflow 里复现 compute 侧 accumulation

---

## 核心运行时类一览

| 类 | 为什么重要 |
|---|---|
| `MPIComputeEngine` | task-scoped MPI runtime 的核心 |
| `TrajectoryCache` | per-rank 的 per-frame observable 存储，可 gather 到 rank 0 |
| `FrameCache` | 轻量级 per-frame 几何 digest |
| `TopologyArrays` | 冻结拓扑快照，broadcast 给所有 worker |
| `Forcefield` | 模型参数和 mask 的快照 |
| `FrameGeometry` | 不可变 per-frame 几何对象（传给 energy/force） |

理解了这六个对象，大部分 runtime 行为就可预测了。
