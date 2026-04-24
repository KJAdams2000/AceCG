# 10 VP Grower 开发者参考

*Updated: 2026-04-23.*

> 本章只覆盖 VP grower 管线。训练 workflow 见 [09_workflows.md](09_workflows.md)。

VP grower 在当前 active repo 里是一条独立的一次性数据生产管线：

- 不继承 `BaseWorkflow`
- 不参与 trainer / optimizer / checkpoint 训练循环
- 不复用 `TaskScheduler.run_iteration()`

它的目标是从 CG-only 参考拓扑和轨迹，生成：

- 带 VP 的 schema 拓扑
- `latent.settings` 和初始 pair/bond/angle tables
- 每个 conditioning frame 的 `frame_*.data`
- 可选的 `frame_*.forces.npy`
- `manifest.json`

---

## 核心文件

| 文件 | 职责 |
|---|---|
| `workflows/vp_growth.py` | 顶层 one-shot orchestrator，决定 universe loading 策略和输出布局 |
| `topology/vpgrower.py` | VP template 构建、单帧 VP 放置、`.data` 写出 |
| `compute/vp_prepare.py` | MPI-parallel frame growth、manifest 汇总 |
| `io/vp_ffbuilder.py` | `latent.settings` 和初始 VP tables 的导出 |
| `configs/vp_growth_config.py` | VP grower 专用配置模型和解析 |

---

## 静态模板 vs 动态帧

VP grower 的设计核心是把“不会变的拓扑模板”和“每帧几何”分开。

| 层次 | 主要对象 | 说明 |
|---|---|---|
| 静态模板 | `VPTopologyTemplate` | atom names、type ids、插入后的 bonds / angles / dihedrals、real/vp index 映射 |
| 单帧几何 | `VPGrownFrame` | 和模板对齐的 `(n_atoms, 3)` 坐标 + box dimensions |
| 执行器 | `VPGrower` | 持有模板，按帧调用 `grow_frame(...)` |

这让 rank 间共享的数据很小：广播模板即可，不需要每次重建拓扑。

---

## 关键入口

| 符号 | 位置 | 作用 |
|---|---|---|
| `VPGrowthWorkflow.run()` | `workflows/vp_growth.py` | 顶层执行：读 config、选 universe 策略、写 schema、分发帧增长 |
| `VPGrower.from_universe()` | `topology/vpgrower.py` | 从 CG-only `MDAnalysis.Universe` 构建 `VPTopologyTemplate` |
| `VPGrower.grow_frame()` | `topology/vpgrower.py` | 给一帧 real-site 坐标插入 VP 坐标 |
| `grow_vp_frames()` | `compute/vp_prepare.py` | 按 frame-id 划分工作，MPI-parallel 写 `frame_*.data` |
| `write_latent_settings()` | `io/vp_ffbuilder.py` | 导出 `latent.settings` 和初始 VP tables |
| `write_vp_data()` | `topology/vpgrower.py` | 把模板 + 单帧几何写成 LAMMPS data |

---

## `VPGrowthWorkflow`

`VPGrowthWorkflow` 是一个 one-shot driver，而不是训练 workflow。

顶层步骤：

1. 读取 `VPGrowthConfig`
2. 判断是广播整条 universe，还是每个 rank 只加载本地 segment
3. rank 0 构建 `VPGrower` / `VPTopologyTemplate`
4. rank 0 写 `vp_topology.data` 和 `latent.settings`
5. 所有 rank 调用 `grow_vp_frames()`
6. rank 0 汇总 `manifest.json`

当前实现里，trajectory loading 策略是显式的：

- segment 数很少时，广播完整 universe
- segment 较多时，每个 rank 只打开自己负责的 trajectory 子集

这套策略直接写在 `workflows/vp_growth.py`，不依赖 scheduler 层。

---

## `VPGrower`

`VPGrower` 自身是“构造后无状态”的执行器。它只持有不可变模板，真正随帧变化的是输入：

- real-site positions
- box dimensions
- orientation seed

`topology/vpgrower.py` 同时拥有：

- VP bond / angle / dihedral spec 的解析结果
- carrier residue 到 VP 插槽的映射
- anti-clash 迭代放置逻辑
- `write_vp_data()` 对 `write_lammps_data()` 的薄包装

设计边界很明确：几何放置只受 VP bonds / angles 和防碰撞逻辑控制；
VP dihedrals 会被写进最终拓扑和 `latent.settings`，但不参与几何 grow 本身。

---

## `grow_vp_frames()`

`compute/vp_prepare.py` 负责并行 frame growth。它的 frame-id 分片策略故意与
`MPIComputeEngine.run_post()` 保持一致，这样 VP grower 和 compute runtime 的
离散帧划分语义是一致的。

输入最重要的几个参数：

- `grower`：共享模板的 `VPGrower`
- `universe`：每个 rank 本地可 seek 的 `MDAnalysis.Universe`
- `frame_ids`：全局 frame-id 列表
- `local_frame_ids`：可选，本 rank 的本地 universe seek index
- `orientation_seed_base`：每帧的随机朝向基准种子

输出约定：

- `frame_{fid:06d}.data`
- 可选 `frame_{fid:06d}.forces.npy`
- rank 0 合并出 `VPGrowManifest`

---

## `write_latent_settings()`

`io/vp_ffbuilder.py` 负责把 VP 模板和 VP config 变成 LAMMPS 可消费的静态文件：

- `latent.settings`
- 初始 pair/bond/angle tables

最重要的两个 helper：

| 函数 | 作用 |
|---|---|
| `build_vp_forcefield()` | 从 `VPConfig + VPTopologyTemplate` 物化 VP-only `Forcefield` |
| `render_vp_latent_template()` | 把最终 `pair_coeff` / `bond_coeff` / `angle_coeff` / `dihedral_coeff` 文本渲染出来 |

这些输出属于 VP grower 管线的一部分，因此不写进通用 IO 章节。

---

## 输出文件

一次 VP grow run 的关键产物通常是：

```
output_dir/
  vp_topology.data
  latent.settings
  Pair_*.table
  Bond_*.table
  Angle_*.table
  frame_000000.data
  frame_000000.forces.npy
  manifest.json
```

其中：

- `vp_topology.data` 是 schema-only 拓扑，供后续 CDREM / CDFM 使用
- `frame_*.data` 是真正每帧 grow 出来的配置
- `manifest.json` 是 rank 0 汇总后的规范索引

---

## 开发规则

如果以后继续扩展 VP grower，优先保持以下边界：

1. 静态拓扑模板继续留在 `topology/vpgrower.py`，不要混进训练 workflow。
2. 并行 frame growth 继续留在 `compute/vp_prepare.py`，保持和 `run_post()` 一致的 frame-id 切分语义。
3. `latent.settings` / VP table 输出继续视为 VP-specific IO，不要混入通用 `io` 开发文档。