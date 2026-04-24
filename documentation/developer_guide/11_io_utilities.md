# 11 IO Utilities 开发者参考

*Updated: 2026-04-23.*

> 本章只覆盖通用 IO utilities。VP-specific `latent.settings` / table 生成见 [10_vp_grower.md](10_vp_grower.md)。

`AceCG.io` 在 active repo 里负责三类事情：

- trajectory / frame extraction
- forcefield / table / coordinate 序列化
- 屏幕日志等轻量辅助工具

它不拥有 trainer 语义、workflow 训练循环或 compute reducer 数学。

---

## 核心模块

| 文件 | 职责 |
|---|---|
| `io/trajectory.py` | 轨迹切分、随机帧读取、one-pass frame iterator |
| `io/forcefield.py` | LAMMPS forcefield / mask 读写 |
| `io/coordinates.py` | AA→CG 坐标映射和 sanity checks |
| `io/coordinates_writers.py` | `.gro` / `.pdb` / LAMMPS `.data` 写出 |
| `io/tables.py` | table 读写、FM table bundle 导出 |
| `io/logger.py` | 屏幕日志和格式化时间戳 |

---

## 轨迹 API

最重要的轨迹入口不是某个 workflow helper，而是 `trajectory.py` 里的几个独立函数。

| 函数 | 作用 |
|---|---|
| `iter_frames(universe, ...)` | compute runtime 和 VP grower 的规范 one-pass frame iterator |
| `read_lammpstrj_frames(path, frame_ids, ...)` | 针对显式 frame-id 列表做随机访问读取 |
| `count_lammpstrj_frames_and_atoms(path)` | 轻量扫描 segment frame 数和 atom 数 |
| `split_lammpstrj(...)` | 基于文本解析切分大轨迹 |
| `split_lammpstrj_mdanalysis(...)` | 基于 MDAnalysis 的切分路径 |

开发者合同：

- `iter_frames()` 是 compute 层的一等入口，返回 `(frame_id, positions, box, forces)`
- `frame_ids` 一旦显式给出，就覆盖 `start/end/every`
- 是否读取 force 由 `include_forces` 和 trajectory 本身是否带 force 列共同决定

---

## Forcefield I/O

`io/forcefield.py` 是 runtime `Forcefield` 与 LAMMPS settings 文件之间的桥。

| 函数 | 作用 |
|---|---|
| `ReadLmpFFMask()` | 解析 authored forcefield mask 文件 |
| `ReadLmpFF()` | 从 LAMMPS-style settings 读出 `Forcefield` |
| `WriteLmpFF()` | 把当前 `Forcefield` 写回新的 settings / table 文件 |
| `resolve_source_table_entries()` | 为 FM source-table 路径解析原始 table token 和 table name |

`ReadLmpFF()` 当前有两个关键约定：

- 若传入 `topology_arrays`，bond/angle type id 会被还原成规范 `InteractionKey`
- `table` 样式不会直接保留原始 table，而是按 `table_fit` 拟合成 runtime potential

`WriteLmpFF()` 的对偶语义是：

- 对非 table 项直接回填参数
- 对 table 项重新生成 table 文件

因此它是 workflow 导出 runtime forcefield bundle 的规范写口。

---

## 坐标构建与 `.data` 写出

这部分最重要的高层入口是：

| 函数 | 作用 |
|---|---|
| `build_CG_coords()` | 从 AA 坐标 + mapping 构建 CG beads，并可选写 `.gro` / `.pdb` / `.data` |
| `write_lammps_data()` | 写带可选 bonds / angles / dihedrals 的 LAMMPS data 文件 |
| `write_gro()` / `write_pdb()` | 轻量结构文件写出 |

`build_CG_coords()` 负责：

- 读取 YAML mapping 或内存 mapping dict
- 做 mapping sanity check
- 通过 AA topology / trajectory 计算 bead 位置和质量
- 按 `outputs={...}` 决定是否立即写文件

`write_lammps_data()` 则是更底层的 writer，当前已经支持：

- `atomic` / `full` atom style
- 可选 bonds / angles / dihedrals 及其 type ids
- triclinic box
- image flags 和 wrapped coordinates

当你已经有坐标、type ids 和拓扑数组时，应直接调用 `write_lammps_data()`；
当你还在做 AA→CG 映射时，用 `build_CG_coords()` 更合适。

---

## Table I/O

`io/tables.py` 同时服务于 forcefield 读写、FM table 导出和表文件比较。

最重要的函数：

| 函数 | 作用 |
|---|---|
| `parse_lammps_table()` | 低层 table parser |
| `write_lammps_table()` | 低层 table writer |
| `build_forcefield_tables()` | 从 FM runtime spec + `Forcefield` 生成内存 payload |
| `export_tables()` | 把整套 FM tables 写到目录并返回 manifest |
| `compare_table_files()` | 比较参考 table 和候选 table 的差异 |
| `cap_table_forces()` | 对 table force 做硬上限裁剪 |

对于 workflow 层，真正的高层入口通常是 `export_tables()`；
`parse_lammps_table()` / `write_lammps_table()` 更多是底层 building block。

---

## Logger

`io/logger.py` 很薄，但几乎所有 workflow / scheduler / compute 入口都会用到它。

最常用的符号：

| 符号 | 作用 |
|---|---|
| `get_screen_logger(name)` | 获取模块级 `ScreenLogger` |
| `format_screen_message(...)` | 统一消息格式 |
| `user_timestamp()` | 用户可读时间戳 |

这层只负责轻量控制台日志，不负责结构化事件采集或文件日志轮转。

---

## 使用规则

为了保持 IO 层稳定，开发时优先遵循这几个规则：

1. one-pass runtime 优先使用 `iter_frames()`，不要在 workflow 里重新实现逐帧读取。
2. runtime forcefield 导出优先使用 `WriteLmpFF()`，不要手工拼 LAMMPS settings。
3. FM table 导出优先使用 `export_tables()`，保持 manifest 和文件名约定一致。
4. VP-specific 的 `latent.settings` / VP table builder 继续留在 [10_vp_grower.md](10_vp_grower.md) 那条独立管线里。