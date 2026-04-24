# 02 Topology 模块开发者参考

*Updated: 2026-04-23.*

拓扑层由三个核心文件构成，外加两个辅助模块：

| 文件 | 职责 |
|---|---|
| `types.py` | `InteractionKey` — 所有相互作用类型的规范化哈希键 |
| `topology_array.py` | `TopologyArrays` — 冻结数据类，从 MDAnalysis Universe 构建一次，广播给 MPI worker |
| `forcefield.py` | `Forcefield` — `InteractionKey → List[BasePotential]` 字典容器，附带 mask、bounds、参数向量缓存 |
| `neighbor.py` | 拓扑感知近邻搜索，输出 `{InteractionKey: (a_idx, b_idx)}` |
| `mscg.py` | MS-CG 拓扑辅助工具（依赖核心三件套）|

---

## `InteractionKey`（types.py）

`NamedTuple(style, types)`。全局用作字典键。

```python
key = InteractionKey.bond("A", "B")
key = InteractionKey.pair("C", "A")     # 自动排序 → ("A","C")
key = InteractionKey.angle("X", "Y", "Z")
key = InteractionKey.dihedral("A", "B", "C", "D")
```

### 规范化规则

所有构造函数产出规范化顺序，使 `key(A,B) == key(B,A)`：

| Style | 规则 | 例子 |
|---|---|---|
| pair | 字母序：`(a,b) if a ≤ b` | `pair("C","A")` → `("A","C")` |
| bond | 同 pair | 同上 |
| angle | 若 `a > c` 则反转：`(a,b,c) if a ≤ c else (c,b,a)` | `angle("Z","Y","A")` → `("A","Y","Z")` |
| dihedral | 若 `(a,b) > (d,c)` 则反转 | `dihedral("D","C","B","A")` → `("A","B","C","D")` |

angle 的中心原子是 `b`。dihedral 的中心键是 `b-c`。

### 序列化

```python
key.label()                           # "bond:A:B"
InteractionKey.from_label("bond:A:B") # 对称反序列化
```

---

## `TopologyArrays`（topology_array.py）

**冻结数据类**，通过 `collect_topology_arrays()` 从 MDAnalysis Universe 构建一次，
然后广播给所有 MPI worker。所有访问通过属性，不用字典键。

### 构建

```python
from AceCG.topology.topology_array import collect_topology_arrays

topo = collect_topology_arrays(
    universe,
    exclude_bonded="111",            # 3字符 flag：包含 1-2、1-3、1-4 排除
    exclude_option="resid",          # nonbonded 排除策略：resid / molid / none
    atom_type_name_aliases={1: "CA", 2: "CB"},  # 可选，LAMMPS type-code → name
    vp_names=["VP"],                 # 可选，虚位点 type 名称
)
```

### LAMMPS alias 协议

`collect_topology_arrays()` 允许合成 atom name：

- 若 `u.atoms.names` 已存在，直接使用
- 若 names 不存在且提供了 `atom_type_name_aliases`，将整数 type code 映射为字符串 name 并写回 Universe
- 若 names 不存在且未提供 alias，复用 `u.atoms.types` 作为 name 写回 Universe

因此 bonded `InteractionKey` 用的名字来源是：LAMMPS 输入用 alias/合成 name；已有 names+types 的拓扑用 `u.atoms.types`。

`atom_type_name_aliases` 的 key 必须是整数型。同一 type id 的冲突 alias 会报错。

### 字段参考

**原子级**

| 字段 | Shape / type | 说明 |
|---|---|---|
| `n_atoms` | int | 总原子数（含虚位点）|
| `names` | `(n_atoms,)` str | per-atom name |
| `types` | `(n_atoms,)` str | per-atom type name |
| `atom_type_names` | `(n_unique,)` str | 有序唯一 type name 列表 |
| `atom_type_codes` | `(n_atoms,)` int32 | 指向 `atom_type_names` 的 1-based 编码 |
| `masses` | `(n_atoms,)` float64 | 原子质量 |
| `charges` | `(n_atoms,)` float64 | 原子电荷 |
| `atom_resindex` | `(n_atoms,)` int64 | 每个原子所在 residue index |
| `molnums` | `(n_atoms,)` int64 | 每个原子的分子编号（absent 时填 0）|

**Residue 级**

| 字段 | Shape / type | 说明 |
|---|---|---|
| `n_residues` | int | 残基总数 |
| `resids` | `(n_residues,)` int64 | 残基 id |

**成键项**

| 字段 | Shape / type | 说明 |
|---|---|---|
| `bonds` | `(n_bond, 2)` int64 | 原子 index 对 |
| `angles` | `(n_angle, 3)` int64 | 原子 index 三元组 |
| `dihedrals` | `(n_dihedral, 4)` int64 | 原子 index 四元组 |

**成键排除列表**（供近邻搜索构建用）

| 字段 | Shape / type | 说明 |
|---|---|---|
| `exclude_12` | `(n_ex, 2)` int64 | 成键 1-2 对 |
| `exclude_13` | `(n_ex, 2)` int64 | angle 1-3 端点 |
| `exclude_14` | `(n_ex, 2)` int64 | dihedral 1-4 端点 |

**预编码排除数组**（供 `neighbor.py` 高效查表用）

| 字段 | Shape / type | 说明 |
|---|---|---|
| `excluded_nb` | `(n_ex,)` int32 | 预编码的排除 pair ID，供快速集合运算 |
| `excluded_nb_mode` | str | 构建时的 `exclude_option`（`"resid"` / `"molid"` / `"none"`）|
| `excluded_nb_all` | bool | 若为 True，表示系统已全局排除（单分子或单 residue），near-neighbor 搜索可直接跳过 |

`excluded_nb` 用 `a * n_atoms + b` 编码 `(a, b)` 对，`neighbor.py` 用 `np.isin()` 做 O(1) 向量化查找。

**虚位点分类**

| 字段 | Shape / type | 说明 |
|---|---|---|
| `real_site_indices` | `(n_real,)` int64 | 非虚位点原子 indices |
| `virtual_site_mask` | `(n_atoms,)` bool | True 表示虚位点 |
| `virtual_site_indices` | `(n_virtual,)` int64 | 虚位点原子 indices |

若未提供 `vp_names`：`real_site_indices = arange(n_atoms)`，`virtual_site_mask = all-False`。

**实例 → 类型 映射**

| 字段 | Shape / type | 说明 |
|---|---|---|
| `bond_key_index` | `(n_bond,)` int32 | 每个 bond 实例 → `keys_bondtypes` 中的 index |
| `angle_key_index` | `(n_angle,)` int32 | 同上，用于 angle |
| `dihedral_key_index` | `(n_dihedral,)` int32 | 同上，用于 dihedral |
| `keys_bondtypes` | `List[InteractionKey]` | bond-type key 列表（按 index 顺序）|
| `keys_angletypes` | `List[InteractionKey]` | angle-type key 列表 |
| `keys_dihedraltypes` | `List[InteractionKey]` | dihedral-type key 列表 |

**类型转换 dict**

| 字段 | 类型 | 说明 |
|---|---|---|
| `atom_type_name_to_code` | `dict[str, int]` | atom type name → int code |
| `atom_type_code_to_name` | `dict[int, str]` | int code → atom type name |
| `bond_type_id_to_key` | `dict[int, InteractionKey]` | bond type index → canonical key |
| `angle_type_id_to_key` | `dict[int, InteractionKey]` | angle type index → canonical key |
| `dihedral_type_id_to_key` | `dict[int, InteractionKey]` | dihedral type index → canonical key |
| `key_to_bonded_type_id` | `dict[InteractionKey, int]` | canonical key → bonded type index |

### 不变量

- **冻结**：构建后不可变。
- 所有数组为稠密 NumPy。空 topology 产出 shape-`(0, width)` 数组，不是 `None`。
- `bond_key_index[i]` 索引 `keys_bondtypes`，两者必须来自同一个 Universe。
- `real_site_indices + virtual_site_indices` 是 `arange(n_atoms)` 的划分。
- `exclude_bonded` 只控制成键不变排除（`exclude_12/13/14`）。nonbonded 排除策略在 `neighbor.py`。
- `excluded_nb_all=True` 时 `excluded_nb` 为空数组（单分子系统的快速路径）。
- `molnums` 源拓扑中不存在时填 0。

---

## `Forcefield`（forcefield.py）

`MutableMapping[InteractionKey, List[BasePotential]]`，附带 mask、bounds、param vector 缓存。

### 构建

```python
from AceCG.topology.forcefield import Forcefield

ff = Forcefield({key: [pot1, pot2]})  # 从 dict 构建
ff2 = Forcefield(ff)                  # copy-construct（shallow potential refs）
ff3 = ff.deepcopy()                   # 完整深拷贝
ff4 = copy.deepcopy(ff3)              # 同上，两者均安全
```

### 参数向量

扁平参数向量是所有 potential 的 `get_params()` 按插入顺序拼接而成。

```python
ff.n_params()          # 总标量参数数
L = ff.param_array()   # (n_params,) float64 拷贝
ff.update_params(L)    # 写回所有 potential + 刷新缓存
```

切片辅助：

```python
ff.param_slices()         # [(key, pot_idx, slice), ...]
ff.interaction_offsets()  # [slice, ...]
ff.param_index_map()      # [(key, "k"), (key, "r0"), ...]
```

### Mask

**L2（per-parameter）** `param_mask` 和 **L1（per-key）** `key_mask` 双向同步：

```python
ff.param_mask                                       # (n_params,) bool，默认全 True
ff.param_mask = np.array([True, True, False])        # setter 推导 key_mask

ff.key_mask                                         # {key: bool}
ff.key_mask = {bond_key: False}                     # setter 传播到 param_mask
```

- L1→L2：设置 `key_mask = {k: False}` 将 key k 的整个 param block 清零
- L2→L1：`key_mask[k] = any(param_mask[block])`，只要有一个参数 active，该 key 就 active

按 glob/regex 模式构建：

```python
ff.build_mask(mode="freeze", global_patterns=["*k*"])
ff.build_mask(mode="train", patterns={key: ["r0"]})
```

### Bounds

```python
lb, ub = ff.param_bounds
ff.param_bounds = (lb, ub)
ff.build_bounds(global_bounds={"*k*": (0, None)})  # 模式式构建
L_safe = ff.apply_bounds(L)
```

有 `param_bounds()` 方法的 potential 在构建时自动填充 bounds。

### VP mask

构建一次，冻结（因为 VP 是预定义的拓扑特性）：

```python
ff.set_vp_masks(["VP"])
ff.virtual_mask         # key 的所有类型都是 VP → True
ff.real_mask            # key 的类型都不是 VP → True
ff.real_virtual_mask    # 含 VP 和 real 的混合 key → True
ff.direct_active_mask   # ~virtual_mask
```

### Key 增删

```python
ff[new_key] = [pot]  # 插入：param vector 增长，cache 局部更新
del ff[old_key]      # 删除：param vector 缩短，cache 局部更新
```

增量 `_splice_caches` 在不全量重建的情况下保持 `param_mask`、`param_bounds`、VP mask 一致。

### 遍历

```python
for key in ff:                           # 遍历 key
for key, pot in ff.iter_potentials():    # 展平遍历 BasePotential
```

---

## `neighbor.py`

`neighbor.py` 是 topology 层的近邻搜索辅助模块。
消耗原始坐标 + 一个 `TopologyArrays` 快照，返回原子 index pair 或邻接表。
**不计算距离、能量、力或 `FrameGeometry`**。

### 公开入口点

| 函数 | 职责 |
|---|---|
| `parse_exclude_option(s)` | 规范化 exclude_option 字符串（支持多种 alias）|
| `compute_pairs_by_type()` | **当前引擎路径**：一次全局近邻搜索，按 canonical `InteractionKey` 分 bin |
| `compute_neighbor_list()` | 通用 per-atom 邻接表，当前 compute core 不使用 |

### 排除协议

成键排除始终来自 `TopologyArrays`：

- `exclude_12` / `exclude_13` / `exclude_14`（构建时已合并入 `excluded_nb`）

额外 nonbonded 排除由 `exclude_option` 选择：

| Option | 含义 |
|---|---|
| `resid` | 排除同 residue pair |
| `molid` | 排除同分子 pair |
| `none` | 不额外排除 |

`parse_exclude_option()` 接受 alias（如 `"residue"` → `"resid"`，`"mol"` → `"molid"`）。

### `compute_pairs_by_type()`

```python
from AceCG.topology.neighbor import compute_pairs_by_type

pair_cache = compute_pairs_by_type(
    positions=positions,
    box=box,
    pair_type_list=pair_keys,
    cutoff=cutoff,
    topology_arrays=topo,
    sel_indices=sel_indices,
    exclude_option="resid",
)
```

返回：

```python
{InteractionKey: (a_idx, b_idx)}   # a_idx, b_idx 为全局原子 index 数组
```

### 边界与不变量

- 一个全局 cutoff（通常取所有 pair potential 的最大截断）
- `sel_indices` 缩小搜索域，但**不**重新编号原子
- 输出 index 始终是全局原子 index
- per-key cutoff 过滤推迟到 `compute_frame_geometry()` 中
- 此模块只做拓扑侧路由，应保持数值上简单

---

## 数据流边界

### 所有权

| 数据 | 所有者 | 消费者 |
|---|---|---|
| `param_mask`（L2）| `Forcefield` | `energy()` / `force()` 通过 `ff.param_mask` 读取 |
| `key_mask`（L1）| `Forcefield` | 同上；engine 传给 `compute_frame_geometry()` 作为 `interaction_mask` |
| `real_site_indices` | `TopologyArrays` | `compute_frame_geometry()`→`FrameGeometry`→`force()` |

### Snapshot 协议

Workflow 在 dispatch 给 engine 前创建 **Forcefield snapshot**，携带当前 mask：

```python
ff_snap = Forcefield(self.trainer.forcefield)   # copy-construct
# 传入 engine/reducers 的 forcefield_snapshot
```

`Forcefield.param_mask` / `key_mask` 是规范的 trainability state。
Optimizer 在初始化时接收 mask 副本用于执行；workflow 在 snapshot 前只更新 `Forcefield` 上的 mask。

Engine 通过 MPI broadcast 传播 `forcefield_snapshot` 和 `topology_arrays`。
所有 per-frame mask 从这两个对象读取，不额外穿透传入。

### L1 `interaction_mask` 在 engine 里的流动

Engine 读取 `forcefield_snapshot.key_mask` 一次，作为 `interaction_mask` kwarg 传给 `compute_frame_geometry()`，跳过禁用 key 的几何计算。
同一个 `key_mask` 在 `energy()` / `force()` 里再次检查，作为安全兜底。

---

## Workflow 合法访问模式

**Do**：

- 为目标 topology 构建一个 Universe
- 每次 topology 变化时调用一次 `collect_topology_arrays()`
- 对 LAMMPS 输入显式传入 `atom_type_name_aliases`
- 把 `TopologyArrays` 作为不可变快照传入 compute / engine 代码
- 让 `neighbor.py` 负责 nonbonded 排除路由

**Do not**：

- 就地 mutate `TopologyArrays` 字段
- 在 per-frame 循环内重建 topology arrays
- 绕过 `exclude_bonded` / `exclude_option` 发明 workflow-local 排除规则
- 把 `frame_id`、trajectory slicing 或 MPI rank 分配当做 topology 职责
- 在 topology snapshot 已构建后重新解释 atom-type alias
