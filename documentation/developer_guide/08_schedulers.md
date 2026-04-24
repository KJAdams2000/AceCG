# 08 AceCG `schedulers/` 开发者参考

*Updated: 2026-04-23. 合并自旧版调度器设计文档。*

> 目标读者：需要理解、调试或扩展 AceCG 调度器子系统的开发人员。

Scheduler 层把 workflow 产出的 TaskSpec 翻译成具体的 MPI 进程启动，
并在一个 iteration 内把多个 xz / zbx 任务流式调度到当前机器或
SLURM allocation 的 CPU 资源上。

它拥有：

- 计算资源（节点 + 每节点的 CPU）的**发现**（`ResourcePool`）
- CPU 的**贪心分配**（`LeasePool`、`Placer`）
- MPI 启动命令的**组装**（`MpiBackend` 家族）
- 任务子进程的**启动、轮询、信号传递、超时处理**（`TaskScheduler`）
- 任务进程里 sim + post 的执行（`task_runner`）

它**不**拥有：

- workflow orchestration（由 `workflows/` 做）
- 任何 trainer / optimizer / reducer 的语义
- 训练统计的跨 iteration / 跨 task 累积

---

## 1. 核心文件

| 文件 | 职责 |
|---|---|
| `schedulers/resource_pool.py` | 节点/CPU 发现，`HostInventory`、`LeasePool`、`Placer`、`ResourcePool` |
| `schedulers/mpi_backend.py` | 数据类型 `HostSlice` / `Placement` / `LaunchSpec` + Intel/OpenMPI/MPICH 三个具体实现 |
| `schedulers/task_scheduler.py` | 流式调度引擎 `TaskScheduler.run_iteration()`、`_stream()`、进程管理 |
| `schedulers/task_runner.py` | 计算节点上的 sim + post 执行器（子进程入口） |
| `schedulers/profiler.py` | 可选的 MPI 预检基准测试（旁路，不影响主流程） |

---

## 2. 调用链总览

```
Workflow
  → _build_resource_pool()              # workflows/base.py
    → ResourcePool.discover()           # 自动发现节点/CPU + 选择 MPI 后端
  → TaskScheduler(pool)
    → run_iteration(xz_tasks, zbx_tasks)
      → Placer.place(cores)             # 从 LeasePool 租借 CPU
      → backend.realize(placement)     # 生成 MPI 命令行
      → subprocess.Popen(...)          # 启动 task_runner
        → task_runner.py main()        # 在计算节点上：运行模拟 + 后处理
```

---

## 3. 三层设计

Scheduler 按"资源发现 → 资源分配 → 命令组装"三层拆开，每层单独可替换：

```
┌──────────────────────────────────────────────────┐
│  Layer 1: 资源发现                                │
│    HostInventory (hostname, cpu_ids)             │
│    ResourcePool.discover()                       │
└────────────────────┬─────────────────────────────┘
                     │ list[HostInventory]
┌────────────────────▼─────────────────────────────┐
│  Layer 2: 资源分配（贪心）                         │
│    LeasePool         (per-CPU free pool)         │
│    Placer.place()    (best-fit + multi-host)     │
│    Placement         (tuple of HostSlice)        │
└────────────────────┬─────────────────────────────┘
                     │ Placement
┌────────────────────▼─────────────────────────────┐
│  Layer 3: 命令组装（vendor-specific）              │
│    MpiBackend.realize(placement, payload)        │
│    LaunchSpec (argv, env_add, env_strip_prefixes)│
└──────────────────────────────────────────────────┘
```

每层的输入/输出是纯数据类（`HostInventory` / `Placement` / `LaunchSpec`），
互不跨层耦合：

- 切换 MPI 实现**只需改 Layer 3**（MpiBackend）。
- 切换资源发现方式（换 YARN / 换 k8s）**只需改 Layer 1**。
- Layer 2 的贪心算法与前后两层没有耦合，可独立测试。

---

## 4. Layer 1：资源发现

### 发现优先级

`ResourcePool.discover()` 按以下顺序决定 `list[HostInventory]`：

1. **`explicit_hosts`**（`.acg` 里手写的节点+CPU 列表） ← **用户覆盖，最优先**
2. **`scontrol show job -d`**（SLURM 环境，无需 SSH）— 解析 `Nodes=xxx CPU_IDs=yyy`
3. **`SLURM_NODELIST` + SSH 探测** — 逐节点 SSH，`os.sched_getaffinity(0)`
4. **`localhost`** — 非 SLURM 环境回退

| 方法 | 延迟 | 可靠性 | 需要什么 |
|---|---|---|---|
| `scontrol show job -d` | <1s | 高（读 SLURM DB） | `SLURM_JOB_ID` + `scontrol` 在 PATH |
| SSH + `sched_getaffinity` | 5-10s/节点 | 中（SSH 可能超时） | 免密 SSH + Python3 |
| `explicit_hosts` | 0s | 最高（用户保证） | 用户知道分配详情 |

### 三种启动环境

所有 backend 在 `realize()` 内部根据 `SLURM_JOB_ID` 和 `_is_local(placement)` 自动分派：

| 环境 | 触发条件 | 典型场景 |
|---|---|---|
| **local** | 所有核心在本机 | 登录节点 smoke test、单节点 sinteractive |
| **multi-host SLURM** | `SLURM_JOB_ID` 存在，且 placement 跨节点 | 生产环境 sbatch / sinteractive 多节点 |
| **multi-host SSH** | `SLURM_JOB_ID` 不存在，且 placement 跨节点 | 有免密 SSH 的小集群或私有机器 |

### `scontrol` 发现优化

早期版本用 `SLURM_NODELIST + SSH` 逐节点探测（12 节点 60–120 s）。
改成 `scontrol show job -d` 一次拉全部分配后，延迟降到 <1 s，无 SSH 开销。
解析格式（`_discover_hosts_scontrol()`，`resource_pool.py:421`）：

```
  Nodes=midway3-0022 CPU_IDs=16-21 Mem=22860 GRES=
  Nodes=midway3-0095 CPU_IDs=7,10-13,18,46 Mem=26670 GRES=
```

CPU_IDs 是逗号分隔的范围列表，由 `_parse_cpu_range()` 解析。

### 为什么需要 `explicit_hosts`

两类情况 `scontrol` 会失效：

1. **Shell 丢失了 SLURM 环境**：通过 `ssh compute_node 'python -m AceCG ...'`
   进入计算节点时，`SLURM_JOB_ID` / `SLURM_NODELIST` 在 SSH 中丢失。
2. **CPU 分配非整节点**：`sinteractive` 拿到碎片化 CPU（每节点 5–8 个），
   SSH fallback 探测到的是整节点 affinity，因为 SSH 进来的 shell 不在 job cgroup 里。

`.acg` 里手写 `explicit_hosts` 是最 robust 的做法。注意：这是写在
`[scheduler]` 段里的原始 list-of-dicts 配置，parser 会把它保存在
`scheduler.extras["explicit_hosts"]`：

```ini
[scheduler]
explicit_hosts = [
  {"hostname": "midway3-0022", "cpu_ids": [16, 17, 18, 19, 20, 21]},
  {"hostname": "midway3-0095", "cpu_ids": [7, 10, 11, 12, 13, 18, 46]},
]
```

---

## 5. Layer 2：贪心资源分配

### 数据结构

- **`LeasePool`**：按 hostname 维护 `{host: sorted list of free cpu_ids}`，
  同时记 `_total[host]`（Intel MPMD 路径需要 `host_total()`）。
- **`CpuLease`**：一次 `acquire()` 产出的块，`(host, cpu_ids)`。
- **`Placer`**：接受 `LeasePool + MpiBackend`，对一个任务产出
  `PlacementResult(placement, leases)`。

### `Placer.place()` 算法

```python
place(min_cores, preferred_cores, max_cores, single_host_only)
```

两条路径：

**单机优先**：从 `preferred_cores` 往下降到 `min_cores`，只要任何
一个 host 的 free ≥ n，就直接 `acquire(n)`。

**多机 fallback**：单机路径失败 **且** `single_host_only=False` **且**
`backend.supports_multi_host=True` 时，走贪心跨节点：

- **`whole_node=False`**（默认，SSH bootstrap / srun 路径）：按 free 数降序贪心，
  每个节点最多拿 `target - collected` 个。
- 旧的 whole-node 约束已经移除；当前 Intel MPMD 路径使用 per-segment
  pinning，不再要求整节点预留。

### `_allocate_contiguous()`

从 free CPU 列表里优先取**连续段**（如 CPU 16–21 整块），避免跨 NUMA 域。
找不到连续段时降级为前 n 个空闲核心。

---

## 6. Layer 3：MPI 命令组装

### 数据流

```
Placement ─┐
payload_cmd ├─→ backend.realize() ─→ LaunchSpec
run_dir    ─┘
```

- `Placement` = 一组 `HostSlice`（host + cpu_ids）
- `payload_cmd` = 应用侧 argv，如 `["lmp"]` 或 `["python", "-m", "AceCG.compute.mpi_engine"]`
- `LaunchSpec` = 可直接 `subprocess.Popen` 的 `(argv, env_add, env_strip_prefixes)`

### Backend 选择

```python
pick_backend(mpirun_path, intel_launch_mode="mpmd")
```

通过 `mpirun --version` 输出自动检测 MPI 族：

| 输出包含 | Backend |
|---|---|
| `"intel"` / `"impi"` | `IntelMpiBackend` |
| `"open mpi"` | `OpenMpiBackend` |
| `"mpich"` / `"hydra"` | `MpichBackend` |
| 其他 | `LocalMpirunBackend`（仅单机） |

### 三个 vendor backend 的 SLURM 路径

| Backend | SLURM 命令 |
|---|---|
| **IntelMPI (mpmd)** | `mpirun -bootstrap slurm -n X -host A -env I_MPI_PIN_PROCESSOR_LIST ... : -n Y -host B ...` |
| **IntelMPI (srun)** | `srun --mpi=pmi2 --overlap --exact --distribution=arbitrary --cpu-bind=map_cpu:c0,c1,...` |
| **OpenMPI** | `srun --mpi=pmi2 --overlap --exact --distribution=arbitrary --cpu-bind=map_cpu:... + SLURM_HOSTFILE` |
| **MPICH** | 同 OpenMPI SLURM 路径 |

三个关键不变量：

1. **Per-rank CPU pinning**：所有 SLURM 路径都用 `map_cpu:c0,c1,c2,...`
   把 rank i 精确 pin 到 `cpu_ids[i]`，不靠 SLURM 的 block/cyclic distribution。
   这是唯一能在异构节点分配下正确 pin 的办法。
2. **`SLURM_HOSTFILE` + `--distribution=arbitrary`**：OpenMPI 和 MPICH 用
   这对组合实现每节点任意 rank 数。hostfile 一行一 rank，`arbitrary` 让
   srun 严格按 hostfile 顺序分配。
3. **`env_strip_prefixes=("PMI_", "SLURM_")`**：进入 `_launch()` 前先把
   父进程残留的 PMI / SLURM step-scoped 变量全剥掉，再 inject 干净的
   `SLURM_JOB_ID` + `SLURM_CONF` + `SLURM_HOSTFILE`。不剥会让嵌套的
   mpirun / srun 误以为自己还在父 step 里。

### Intel MPI 的 mpmd vs srun

这是整个 scheduler 最绕的一块，单独列出。

**MPMD 模式**（默认，`intel_launch_mode = mpmd`）：

- `mpirun -bootstrap slurm` + MPMD colon 语法
- 对每一个 slice 写 `-n N -host H -env I_MPI_PIN_PROCESSOR_LIST cpu_ids`，用 `:` 连起来
- Hydra 内部自己起 srun step，**不依赖 libpmi2.so**
- **关键**：segment 必须按 `SLURM_JOB_NODELIST` 的 canonical 顺序排序。
  Hydra 的 slurm bootstrap 只起一个 srun step，SLURM 按 job nodelist 的顺序
  做 block 分配；若 segment 顺序与 `SLURM_JOB_NODELIST` 不一致，rank 会错位。
  `_realize_slurm_mpmd()` 里通过 `_sort_slices_by_slurm_nodelist(...)`
  完成这个排序。

**srun 模式**（`intel_launch_mode = srun`）：

- 直接 `srun --mpi=pmi2`，Intel MPI 的 PMI2 client 必须能 `dlopen libpmi2.so`
- 需要 `I_MPI_PMI_LIBRARY` 指向 `libpmi2.so`（`module load slurm/current`）
- 这条路径已经由 live scheduler 明确支持，并且最近的 Intel `srun`
  workflow runs 已经跑通
- 优点：更轻量（少一层 Hydra bootstrap），精确的 per-rank CPU 绑定

`.acg` 配置：

```ini
[scheduler]
mpirun_path = /software/intel/oneapi_hpc_2023.1/mpi/2021.9.0/bin/mpirun
intel_launch_mode = srun    # 或 "mpmd"（默认）
```

### `LaunchSpec.env_add` vs `ResourcePool.extra_env`

| 字段 | 来源 | 作用域 | 例子 |
|---|---|---|---|
| `LaunchSpec.env_add` | Backend 自己生成 | 单次 `realize()` | `I_MPI_PIN_PROCESSOR_LIST`, `SLURM_HOSTFILE` |
| `ResourcePool.extra_env` | `.acg` 里的 `scheduler.extras.extra_env` | 所有 task 子进程 | `FI_PROVIDER=tcp`, `LD_LIBRARY_PATH` 补丁 |

合并顺序：`os.environ → strip → env_add → extra_env`，所以 `extra_env` 能覆盖 backend 选择。

**注意**：`extra_env` **只进子进程**，scheduler 主进程看不到它。
任何"选 backend / 选 launch mode"的逻辑**必须走 config 字段**，不要走 `extra_env`。

---

## 7. 任务流式调度

### `TaskScheduler.run_iteration(xz_tasks, zbx_tasks)`

```
run_iteration(xz, zbx):
  1. 从 ResourcePool rebuild 一个 LeasePool（每 iteration 都是新的）
  2. xz_tasks + zbx_tasks 合并成 pending（xz 在前，优先拿大块 CPU）
  3. Fail-fast：single_host_only task 的 min_cores > max_free_on_any_host → 立即 RuntimeError
  4. _fill()：对 pending 里每个 task 调 placer.place()，能 place 的立即 _launch()，放进 active
  5. 主循环轮询：proc.poll() 每 0.5 s 一次：
       完成 → 释放 lease → 记录 TaskResult → 再调 _fill()
       xz 失败 → _kill_all(active) 杀所有子进程，提前返回
       zbx 失败 → 丢弃这个 replica，继续
  6. 返回后按 min_success_zbx 判定 iteration 是否失败
```

### `_stream()` 内部流程图

```
pending = [xz_task, zbx_0, zbx_1, ..., zbx_K]
active  = {}     # pid → (proc, task, placement_result, t0)

_fill()  ←─────────────────────────────────────┐
  for task in pending:                          │
    pr = placer.place(...)                      │
    if pr is not None:                          │
      _launch(task, pr)                         │
      active[pid] = (proc, task, pr, t0)        │
      pending.remove(task)                      │
                                                │
while active:                                   │
  for pid, (proc, task, pr, t0) in active:     │
    ret = proc.poll()                           │
    if ret is None:                             │
      if time.monotonic() - t0 > timeout:      │
        _timeout_kill(pgid)                     │
      continue                                  │
    release leases(pr.leases)                   │
    record TaskResult(task, ret, ...)           │
    del active[pid]                             │
    if task.is_xz and ret != 0:                 │
      _kill_all(active)                         │
      return FAILED                             │
    _fill() ─────────────────────────────────── ┘
  if no progress: sleep(0.5)
```

### 子进程启动流程 (`_launch`)

```python
def _launch(task, pr):
    # 1. 序列化 task_spec.json
    spec_dict = task.to_spec_dict(cpu_cores)

    # 2. backend.realize() → LaunchSpec (argv + env)
    sim_launch  = backend.realize(placement, sim_cmd, run_dir)
    post_launch = backend.realize(post_placement, py_cmd, run_dir)

    # 3. 构造子进程环境
    env = os.environ - strip_prefixes + env_add + pool.extra_env

    # 4. subprocess.Popen(["python", "-m", "AceCG.schedulers.task_runner", spec.json],
    #                      start_new_session=True)
```

### 进程管理

每个 task 子进程用 `start_new_session=True` 独立进程组，scheduler 跟踪 `pgid`：

- **超时**：`os.killpg(pgid, SIGTERM)` → 5 s 后 `SIGKILL`
- **scheduler SIGTERM**：`_sigterm_handler` 先 killpg 所有 active pgid，再传给 previous handler
- **atexit**：进程退出前也会 killpg 兜底，避免 orphan mpirun

---

## 8. task_runner.py（计算节点入口）

在**计算节点**上作为子进程运行。读 `task_spec.json`，执行：

1. **模拟阶段** — `subprocess.Popen(sim_launch.argv, env=...)` → 运行 LAMMPS
2. **后处理阶段** — 可选，两种模式：
   - `mode="python"` — 本地 Python 函数（bonded projectors、gradient 计算等）
   - `mode="mpi"` — 通过 `post_launch.argv` 启动 MPI 并行后处理

输出 `timing.json`：

```json
{"sim_wall": 340.1, "post_wall": 17.4, "total_wall": 357.5}
```

---

## 9. 常见陷阱与调试指南

### 陷阱 1：不 `module load lammps` 导致环境混乱

**症状**：`srun --mpi=pmi2` 报 `libpmi2.so: cannot open shared object file`，
或 Intel MPI 报找不到 `libfabric.so.1`、`libmpi.so.12`。

**根因**：`module load lammps/<ver>` 同时 prepend PATH（`mpirun`、`lmp`）
和 LD_LIBRARY_PATH（`libmpi.so.*`、`libfabric.so.1`）。不 load 时，
`shutil.which("mpirun")` 可能找到 conda env 里的 MPICH shim，
但 `lmp` 是 Intel MPI 编译的，rank 起不来。

**修复**：

```bash
module load lammps/29Aug2024
source /software/.../conda.sh
conda activate acg       # 最后 activate，避免 conda bin/ 压掉 lammps MPI
hash -r                  # 刷新 bash PATH 缓存
```

**绝对不要**用 `ssh compute_node "source env.sh && python ..."` 这种
fire-and-forget 方式 — SSH 进来的 shell 丢失 `SLURM_JOB_ID` / PMI 环境。

### 陷阱 2：`srun --overlap` 没加 `--exact` 导致序列化

**症状**：多个并发 srun 步骤互相等待，吞吐量下降到 1 任务串行。

**根因**：没有 `--exact` 时，每个 srun 步骤声称占有所有已分配的 task slot。

**修复**：始终用 `srun --overlap --exact`。

### 陷阱 3：重跑时旧进程孤立导致轨迹损坏

**症状**：轨迹文件 `.lammpstrj` 损坏，MDAnalysis 解析报 `IndexError`；
新任务异常缓慢（与旧 LAMMPS 进程争抢相同核心）。

**根因**：kill 主进程后，Hydra 启动的子 LAMMPS 进程没有全部终止。
新 workflow 重跑时复用同一个 `output_dir`，新旧两个 LAMMPS 进程同时
写同一个轨迹文件，帧数据交错（实际案例：2026-04-19，allocation 48784328）。

**修复**：

```bash
# 彻底杀死旧进程
pkill -u $USER -f "task_runner.*cdrem_t5"
pkill -u $USER -f "lmp.*in\.(xz|zbx)\.lmp"
# 或 scancel $SLURM_JOB_ID 杀掉该 job 下所有 srun 步骤

# 清理旧输出目录
rm -rf experiments/<level>/<run_id>/iter_0000/
```

**给开发者的建议**：task_runner 应当在启动 LAMMPS 前检测 run_dir 中是否
已有 `.lammpstrj` 或 `sim.log`，若存在则先删除或报错拒绝启动。

### 陷阱 4：`shutil.which` 在 HPC 文件系统上很贵

**症状**：每个 task realize 慢 2–5 s，iteration 调度开销 >10 s。

**修复**：所有 backend 在 `__init__()` 里缓存 `_srun_path` 和 `_slurm_conf`，
realize 路径只查 self，不再调 `shutil.which()`。

### 陷阱 5：Intel MPMD segment 未按 SLURM node order 排序

**症状**：`mpirun -bootstrap slurm` 的 rank 号与期望的 CPU pinning 错位。

**修复**：`_realize_slurm_mpmd` 里确保
`sorted_slices = _sort_slices_by_slurm_nodelist(placement.slices)`。

### 陷阱 6：`sinteractive --ntasks=N` 导致 CPU cap

**症状**：`--ntasks=64` 拿到的 job 有 `CPUs/Task=1`，step 级 `--cpus-per-task=M`
被 SLURM silently cap 到 1，所有 rank 挤在一个核上。

**修复**：正确提交方式：`sinteractive --nodes=N --ntasks-per-node=M`。

### 陷阱 7：`dict_values` 不可 pickle（Python 3.13）

**症状**：MPI broadcast 在 `topology_array.py` 中失败。

**修复**：用 `list(d.values())` 包裹。

### 调试步骤（通用）

当 task launch 失败、rank 卡住、或 CPU pinning 不对时，按以下顺序排查：

```bash
# 1. 确认 shell 状态正确
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "which mpirun: $(which mpirun)"
mpirun --version

# 2. 手动测试 scontrol 发现
scontrol show job -d $SLURM_JOB_ID | grep -E "Nodes=|CPU_IDs="

# 3. 测试单节点 local 启动
python -c "
from AceCG.schedulers.resource_pool import ResourcePool
pool = ResourcePool.discover()
print(pool.hosts)
"

# 4. 验证 CPU pinning（在计算节点上）
mpirun -n 4 bash -c 'cat /proc/$$/status | grep Cpus_allowed_list'

# 5. 检查 env strip 是否正确
python -c "
from AceCG.schedulers.mpi_backend import pick_backend
b = pick_backend('mpirun')
print(type(b).__name__, b.supports_multi_host)
"
```

---

## 10. 添加新 MPI 后端

1. **在 `mpi_backend.py` 新增 class**：

   ```python
   class FooMpiBackend(MpiBackend):
       name = "foo"
       supports_multi_host = True

       def __init__(self, mpirun_path: str) -> None:
           self.mpirun_path = mpirun_path
           self._srun_path = shutil.which("srun")   # 缓存，避免重复搜 PATH
           self._slurm_conf = _find_slurm_conf()    # 缓存

       def realize(self, placement, payload_cmd, run_dir) -> LaunchSpec:
           if _is_local(placement):
               return self._realize_local(placement, payload_cmd)
           if os.environ.get("SLURM_JOB_ID"):
               return self._realize_slurm(placement, payload_cmd, run_dir)
           return self._realize_ssh(placement, payload_cmd, run_dir)
   ```

2. **三个 `_realize_*` 方法**必须：
   - 产出正确的 argv（per-rank CPU pinning，不依赖 SLURM distribution）
   - `env_add` 里塞上 vendor-specific var
   - `env_strip_prefixes=("PMI_", "SLURM_")`（SLURM 路径）或 `("PMI_",)`（local/ssh）

3. **在 `detect_mpi_family()` 加识别串**，grep `mpirun --version` 输出，返回 `"foo"`。

4. **在 `pick_backend()` 加分派**：

   ```python
   if family == "foo":
       return FooMpiBackend(mpirun_path)
   ```

5. **补单元测试**（`tests/test_scheduler_resource_pool.py` /
   `test_scheduler_streaming.py`）。用 stub backend（`realize()` 返回
   `echo <...>`）验证 scheduler 流程，不必真跑 MPI。

6. **生产 smoke test**：在 `scripts/` 加 `smoke_foo.sh`，跑 3 epoch
   CDREM，对照已有 backend 的 grad_norm 曲线。

---

## 11. 配置参数速查

| `.acg` 字段 | 类型 | 说明 |
|---|---|---|
| `scheduler.launcher` | deprecated | 已忽略；backend 现在由 `scheduler.mpirun_path`、`scheduler.mpi_family` 或 PATH 自动决定 |
| `scheduler.mpirun_path` | `str` | mpirun 绝对路径（留空自动检测） |
| `scheduler.mpi_family` | `str` | 可选显式 backend 覆盖，如 `intel` / `openmpi` / `mpich` |
| `scheduler.python_exe` | `str` | 计算节点上的 Python（默认 `"python"`） |
| `scheduler.task_timeout` | `int` | 任务超时秒数（**必须设置**） |
| `scheduler.min_success_zbx` | `int` | ZBX 最低成功数（默认=全部） |
| `scheduler.explicit_hosts` | `list[dict]` | 写在 `[scheduler]` 段里，运行时保存在 `scheduler.extras.explicit_hosts`，用于手动指定节点+CPU |
| `scheduler.extra_env` | `dict` | 写在 `[scheduler]` 段里，运行时保存在 `scheduler.extras.extra_env`，注入到所有 task 子进程 |
| `scheduler.intel_launch_mode` | `str` | 写在 `[scheduler]` 段里，运行时保存在 `scheduler.extras.intel_launch_mode`；Intel MPI SLURM 模式：`"mpmd"` / `"srun"` |
