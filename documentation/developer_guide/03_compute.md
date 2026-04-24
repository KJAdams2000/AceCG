# 03 Compute 模块开发者参考

*Updated: 2026-04-23.*

compute 层是 trajectory/topology I/O 与 trainers 之间的 task-scoped 数值运行时。

它拥有：

- `FrameGeometry`
- `MPIComputeEngine`（含 `FrameCache`、`TrajectoryCache`；legacy aliases
    `FrameObservables`、`TrajectoryObservablesCache`）
- `energy()` / `force()` 内核
- 有状态 reducer pipeline（`reducers.py`）

它不拥有：

- scheduler 策略
- workflow orchestration
- 跨 task 的 trainer tally
- optimizer 策略

---

## 核心文件

| 文件 | 职责 |
|---|---|
| `compute/frame_geometry.py` | 不可变 per-frame 几何视图 |
| `compute/mpi_engine.py` | task runtime：MPI broadcast、帧提取、post dispatch、observable cache |
| `compute/energy.py` | `energy()` 内核 |
| `compute/force.py` | `force()` 内核 |
| `compute/reducers.py` | 有状态 pipeline reducer（init/consume/finalize，per step_mode） |
| `compute/registry.py` | `build_default_engine()`，注册核心 observable |

---

## 三个持久 Public API

```python
from AceCG.compute import compute_frame_geometry, energy, force

geom = compute_frame_geometry(positions, box, topology_arrays, interaction_mask=None)

energy_dict = energy(
    geom, forcefield,
    return_value=True,
    return_grad=True,
    return_hessian=False,
    return_grad_outer=False,
)

force_dict = force(
    geom, forcefield,
    return_value=True,
    return_grad=True,
    return_hessian=False,
    return_fm_stats=False,
)
```

其他所有 compute 符号都是内部 reducer、params helper 或 post-process 桥接，
不视为稳定公开 API。

---

## `MPIComputeEngine`

`MPIComputeEngine` 是计算运行时对象。

一个 engine 实例拥有：

- 一个 MPI communicator（`comm`，可为 None）
- 一个已注册 observable 的 registry
- `serial_threshold`（帧数低于此值时不并行）

**两个持久公开入口**：

```python
engine = build_default_engine(comm=comm)

# 1. 本地 single-frame 计算（不做 MPI）
result = engine.compute(
    request,              # {need_energy_grad: bool, ...}
    frame,                # (frame_id, positions, box, forces)
    topology_arrays,
    forcefield_snapshot,
    frame_weight=1.0,
    ...
)

# 2. 分布式 post-processing
engine.run_post(spec)     # 见 04_mpi_runtime.md
```

**`engine.compute()`** 是本地的，不做 MPI：

- 接受已本地化的 `frame` 元组
- 构建 `FrameGeometry`
- 评估注册的 observable（按 request 过滤）
- 返回 dict（含 `frame_observables`，如果 `return_observables=True`）

**`engine.run_post(spec)`** 是 scheduled-task 边界：

- rank 0 加载 forcefield snapshot、MDAnalysis Universe、TopologyArrays
- broadcast 给所有 rank
- 每个 rank 分到连续帧或 discrete_ids 的均分段
- 对 one-pass steps 调用 reducer pipeline
- MPI reduce → rank 0 finalize → pickle 写出
- 详见 [04_mpi_runtime.md](04_mpi_runtime.md)

---

## Observable Cache：`FrameCache` 和 `TrajectoryCache`

`run_post()` 支持 per-frame 几何 digest 缓存，用于需要几何信息的下游分析：

```python
spec["collect_observables"] = True   # 每帧构建 FrameCache
spec["gather_observables"]  = True   # 在 MPI 场景下 gather 到 rank 0
spec["observables_output_file"] = "cache.pkl"
```

| 类 | 位置 | 内容 |
|---|---|---|
| `FrameCache` | `compute/mpi_engine.py` | 一帧的轻量几何 digest：pair distances、bond lengths、angles、dihedrals、box |
| `TrajectoryCache` | `compute/mpi_engine.py` | 一个 trajectory 上所有已收集帧的 `FrameCache` 集合 |
| `geometry_to_observables()` | `compute/mpi_engine.py` | 从 `FrameGeometry` 切出 `FrameCache` |

旧名字 `FrameObservables` / `TrajectoryObservablesCache` 仍保留为 compatibility
aliases，但新的文档和新代码应优先使用 `FrameCache` /
`TrajectoryCache`。

这套机制让 `analysis/rdf.py` 可以在不重新读轨迹的情况下访问已计算的几何信息。

---

## Reducer Pipeline

`reducers.py` 为每个 `step_mode` 提供五函数 pipeline：

```python
state = init_*_state(step)           # 初始化 accumulator
req   = request_*(step)              # 告诉 engine 需要哪些 observable
consume_*_frame(state, frame_result) # 每帧更新 state（rank-local）
partials = local_partials_*(state)   # 提取本地 partial（供 comm.reduce）
plan = reduce_plan_*(step)           # {key: "sum"|"gather"} 声明 reduce 策略
result = finalize_*_root(state)      # rank 0 最终化，产出 pickle payload
```

当前 4 套 reducer：

| step_mode | init 函数 |
|---|---|
| `fm` | `init_fm_state` |
| `rem` / `cdrem` | `init_rem_state` |
| `cdfm_zbx` | `init_cdfm_zbx_state` |
| `rdf` | 直接走 `analysis.rdf.interaction_distributions()`（不进 pipeline）|

> `cdfm_y_eff` step mode 已废弃。`y_eff` 现在由 `cdfm_zbx` 在 `run_post()` 预处理阶段
> 从 per-replica 的 init 构型 + 配对的 `.forces.npy` 参考力文件、在 rank 0 上一次性计算
> `y_eff = y_ref - f_theta_cg_only(R_init)` 并广播给所有 ranks；mask 在计算时临时切到
> CG-only，完成后恢复。

**Reducer 约定**：

- 只做本地数学，调用 `engine.compute()` 获取 observable
- 不读轨迹，不做 MPI broadcast，不写文件
- MPI reduce 统一由 `run_post()` 执行

---

## Leaf Step 输出合同

所有 scheduled compute 输出都是 pickle 文件，payload 为一个顶层 dict，
值为 ndarray 和简单标量，无混合格式。

### `step_mode="rem"` / `"cdrem"`

| key | 含义 |
|---|---|
| `energy_grad_avg` | 加权平均 ⟨dU/dθ⟩ |
| `n_frames` | 贡献帧数 |
| `weight_sum` | 总帧权重 |
| `d2U_avg` | （可选）加权平均 Hessian |
| `grad_outer_avg` | （可选）梯度外积均值 |
| `energy_grad_frame` | （可选）per-frame 梯度栈 |

### `step_mode="fm"`

| key | 含义 |
|---|---|
| `JtJ` | 归一化加权 FM 法矩阵 |
| `Jty` | 归一化加权 FM 右侧向量 |
| `y_sumsq` | 归一化加权目标力范数平方 |
| `Jtf` | 归一化加权 J^T f |
| `f_sumsq` | 归一化加权模型力范数平方 |
| `fty` | 归一化加权 f^T y |
| `nframe` | 贡献帧数 |
| `weight_sum` | 总帧权重 |
| `n_atoms_obs` | 每帧观测原子数 |

### `step_mode="cdfm_zbx"`

必需 spec keys：

| key | 含义 |
|---|---|
| `init_force_path` | 与本 replica init 构型配对的参考力 `.npy`，形状 `(n_real, 3)` 或 `(3*n_real,)` |
| `init_frame_id` | init 构型的 frame id（用于 `FrameCache.frame_idx` 标注，默认 0）|
| `mode` | `"direct"` 或 `"reinforce"` |
| `beta` | reinforce 模式的温度因子（direct 模式可省） |

输出 payload：

| key | 含义 |
|---|---|
| `grad_direct` | direct gradient contribution |
| `grad_reinforce` | REINFORCE gradient contribution |
| `sse` | sum squared error |
| `obs_rows` | observation row count |
| `n_samples` | sample count |
| `rmse` | `sqrt(sse / obs_rows)` |

---

## `multi` 模式

`spec["steps"]` 里有多个 step 时，所有 one-pass steps 共享同一次轨迹遍历：

```
run_post(spec with multiple steps):
  shared context 准备一次
  local frames 提取一次
  ┌────────────────────────────────────┐
  │  for each frame:                   │
  │    result = engine.compute(...)   │
  │    consume_step0_frame(state0, .) │
  │    consume_step1_frame(state1, .) │
  │    ...                             │
  └────────────────────────────────────┘
  for each step:
    comm.reduce(local_partials)
    finalize_*_root()
    pickle output
```

**合同**：

- `steps` 里的所有 one-pass step 必须在同一个 trajectory/topology/context 下
- 如果需要不同 trajectory 或 geometry context，用不同 task
- `rdf` steps 不进 one-pass pipeline，在所有 one-pass steps 完成后单独运行

---

## 帧权重语义

`spec["frame_weight"]` 总是 task-local 轨迹权重，在 scheduled task 内部消耗。

适用于所有 step_mode。

**不是**跨 task 权重（如 CDREM `x_weight`、CDFM `x_weight`），
那些属于 trainer batch schema，由 workflow/trainer 处理。

---

## Workflow 合法访问模式

**Do**：

- 构建一个 task spec 然后交给 `MPIComputeEngine.run_post()`
- 当几个 leaf reduction 共享同一 trajectory/topology/context 时，用 `multi`（多个 steps）
- 消费 pickle payload，组装 trainer / solver batch
- 把 `engine.compute()` 只当本地 observable helper

**Do not**：

- 把 reducers 当做主要生产接口直接调用
- 在 workflow 里拥有 MPI broadcast、MPI reduction、trajectory slicing 或 cache lifetime
- 因为 legacy helper 函数存在就在 workflow 里重新实现 compute 侧 accumulation

---

## 必须 task spec keys

| key | 含义 |
|---|---|
| `work_dir` | task 工作目录 |
| `forcefield_path` | forcefield pickle 路径 |
| `topology` | topology 文件路径 |
| `trajectory` | trajectory 路径（str）或路径列表（list[str]）|
| `steps` | list，每个 step 必须有 `step_mode` 和 `output_file` |

常用可选 keys：

| key | 含义 |
|---|---|
| `trajectory_format` | MDAnalysis format string，默认 `LAMMPSDUMP` |
| `frame_start` / `frame_end` | 帧范围，inclusive / exclusive |
| `every` | 采样间隔，默认 1 |
| `frame_ids` | 离散帧选择 |
| `frame_weight` | task-local 帧权重 |
| `sel` | atom 选择表达式，默认 `"all"` |
| `cutoff` | nonbonded 截断 |
| `exclude_option` | pair 搜索排除模式 |
| `exclude_bonded` | bonded 排除 mask，如 `"111"` |
| `atom_type_name_aliases` | atom type int → name 映射 |
| `vp_names` | 虚位点名称 |
| `collect_observables` | bool，是否收集 FrameCache |
| `gather_observables` | bool，是否 gather 到 rank 0 |
| `observables_output_file` | observable cache pickle 输出路径 |
