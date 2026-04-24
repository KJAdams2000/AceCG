# 01 Potential 模块开发者参考

*Updated: 2026-04-23.*

Potential 是建模栈最底层的标量相互作用模型。
每次评估一个相互作用坐标，暴露参数导数供 `compute/`、trainer 和 solver 使用。
**不了解** MPI、`FrameGeometry`、原子所有权或 workflow。

---

## 核心模块

| 文件 | 职责 |
|---|---|
| `potentials/base.py` | `BasePotential` + `IteratePotentials()` |
| `potentials/harmonic.py` | 谐振 bonded / angle potential |
| `potentials/gaussian.py` | 归一化 Gaussian pair potential |
| `potentials/lennardjones.py` | Lennard-Jones 12-6 pair potential |
| `potentials/lennardjones96.py` | Lennard-Jones 9-6 pair potential |
| `potentials/lennardjones_soft.py` | 软核 LJ 变体 |
| `potentials/bspline.py` | force-basis B-spline potential |
| `potentials/multi_gaussian.py` | 归一化 multi-Gaussian family |
| `potentials/unnormalized_multi_gaussian.py` | 非归一化 LAMMPS 风格 multi-Gaussian |
| `potentials/srlrgaussian.py` | SR/LR Gaussian pair potential |

`AceCG.potentials.__init__` 还定义了 `POTENTIAL_REGISTRY`，
即 LAMMPS 风格名称到 concrete potential 类的映射。

---

## 层边界

```
topology / Forcefield
  → key 排序、mask、bounds、参数拼接

potential
  → value(r)、force(r)、关于局部参数的导数通道

compute
  → 通过 Forcefield 和 FrameGeometry 调用 potential，
    组装 per-frame 能量、力、Jacobian、Hessian
```

**Potential 应该拥有**：

- 一个相互作用模型的局部参数向量
- 标量评估规则
- 参数导数通道
- 可选局部 bounds 和线性性元数据

**Potential 不应该拥有**：

- 原子 index
- MPI 或帧迭代
- pair 搜索
- Cartesian 力组装
- workflow 或 trainer 状态

---

## 公开 Potential 接口

`AceCG.potentials` 导出：

| 符号 | 含义 |
|---|---|
| `BasePotential` | 抽象基类 |
| `IteratePotentials` | 展平 `(key, potential)` 迭代辅助 |
| `BSplinePotential` | force-basis spline |
| `GaussianPotential` | 归一化 Gaussian |
| `HarmonicPotential` | 谐振 |
| `LennardJonesPotential` | LJ 12-6 |
| `LennardJones96Potential` | LJ 9-6 |
| `LennardJonesSoftPotential` | 软核 LJ |
| `MultiGaussianPotential` | 归一化 multi-Gaussian |
| `UnnormalizedMultiGaussianPotential` | 非归一化 multi-Gaussian |
| `SRLRGaussianPotential` | SR/LR Gaussian |
| `POTENTIAL_REGISTRY` | style 名称 → 类 查找表 |

---

## `BasePotential`（potentials/base.py）

### 必须实现的方法

| 方法 | 含义 |
|---|---|
| `value(r)` | 在局部坐标 `r` 处的标量势能 |
| `force(r)` | 标量力 $F = -dU/dr$ |

`r` 是局部标量坐标向量，**不是** Cartesian 坐标：

- pair potential：pair 距离
- bond potential：键长
- angle potential：角度坐标
- dihedral：二面角坐标

### 常见元数据

子类需要填充：

| 字段 | 含义 |
|---|---|
| `_params` | 当前参数向量 |
| `_param_names` | 参数标签 |
| `_dparam_names` | 能量一阶导数通道名称 |
| `_d2param_names` | 能量二阶导数通道名称 |
| `_df_dparam_names` | 力一阶导数通道名称 |
| `_param_linear_mask` | per-parameter 线性性 mask |
| `_params_to_scale` | 受 `get_scaled_potential(z)` 影响的参数 |

### 常见辅助方法

| 方法 | 含义 |
|---|---|
| `get_params()` / `set_params()` | 参数访问 |
| `n_params()` | 局部标量参数数 |
| `param_names()` | 参数标签 |
| `energy_grad(r)` | 叠加的 $dU/d\theta$ 通道 |
| `force_grad(r)` | 叠加的 $dF/d\theta$ 通道 |
| `basis_values(r)` | per-parameter 力基函数值 |
| `basis_derivatives(r)` | 力基函数关于 `r` 的导数 |
| `is_param_linear()` | 线性性 mask |
| `get_scaled_potential(z)` | 返回选定参数缩放后的副本 |

---

## 科学合同

### 标量力约定

所有 potential 返回标量力：

$$F(r) = -\frac{dU}{dr}$$

Compute 层通过 `FrameGeometry` 中的几何方向向量将标量量投影回 Cartesian 原子力。

### 能量导数通道

`compute.energy` 由以下构建：

- `energy`
- `energy_grad`（对参数的一阶导数）
- `energy_hessian`（对参数的二阶导数）

分别来自 `value(r)`、`_dparam_names`、`_d2param_names`。

### 力导数通道

`compute.force` 和 FM/CDFM 使用 `force_grad(r)` 或 `basis_values(r)` 构建力 Jacobian。

为了性能，新 potential 应通过 `_df_dparam_names` 或直接覆盖 `basis_values(r)` 提供解析导数。
若不提供，`BasePotential.force_grad()` 退回有限差分——正确但**慢得多**，只应视为开发退路。

---

## 内置 Potential 家族

### 解析参数化 potential

| 类 | 说明 |
|---|---|
| `HarmonicPotential` | 两参数 `k, r0`；`k` 是线性的，`r0` 是非线性的 |
| `GaussianPotential` | 幅度是线性的；中心和宽度是非线性的 |
| `LennardJonesPotential` | epsilon 是线性的，sigma 是非线性的 |
| `LennardJones96Potential` | 与 LJ 12-6 同高层结构 |
| `LennardJonesSoftPotential` | 软核非线性 LJ family |

这些类暴露显式解析的一阶和二阶导数。

### Force-basis potential

| 类 | 说明 |
|---|---|
| `BSplinePotential` | 系数直接参数化力；所有通道是线性的 |

`BSplinePotential` 特殊性：

- 力是直接模型输出
- 能量通过对基函数积分获得
- 所有参数是线性优化通道
- 为 FM/CDFM 提供稠密和稀疏基函数访问

### Multi-component Gaussian family

| 类 | 说明 |
|---|---|
| `MultiGaussianPotential` | 归一化 Gaussian mixture；幅度线性，中心/宽度非线性 |
| `UnnormalizedMultiGaussianPotential` | 非归一化 LAMMPS 风格混合 |
| `SRLRGaussianPotential` | 专用于短程/长程模型的 Gaussian family |

这些模型依赖向量化动态导数分发，暴露组件内二阶导数，组件间二阶导数为零。

---

## `IteratePotentials(forcefield)`

展平迭代辅助：

```python
for key, pot in IteratePotentials(forcefield):
    ...
```

同时适用于：

- 旧式 `Dict[key, BasePotential]`
- 当前式 `Forcefield` / `Dict[key, List[BasePotential]]`

想要展平按参数顺序遍历时用它；想要 grouped-by-key 视图时用 `forcefield.items()`。

---

## Registry 合同

`POTENTIAL_REGISTRY` 将外部 style 名称映射到类：

| Registry key | 类 |
|---|---|
| `harmonic` | `HarmonicPotential` |
| `gauss/cut`, `gauss/wall` | `GaussianPotential` |
| `lj/cut` | `LennardJonesPotential` |
| `lj96/cut` | `LennardJones96Potential` |
| `lj/cut/soft` | `LennardJonesSoftPotential` |
| `table` | `MultiGaussianPotential` |
| `double/gauss` | `UnnormalizedMultiGaussianPotential` |
| `srlr_gauss` | `SRLRGaussianPotential` |

添加新的可序列化 potential family 时，在此处注册。

---

## 与 Forcefield 的交互

Potential 不管理全局 mask 或 bounds，那是 `Forcefield` 的职责。

**Potential 应该提供**：

- 局部参数向量
- 局部导数通道
- 可选 `param_bounds()` 用于局部 bounds
- 通过 `_param_linear_mask` 提供线性性信息

**`Forcefield` 在上层构建**：

- 全局参数向量
- per-parameter 和 per-key mask
- 全局 bounds 数组
- 参数切片和偏移

---

## 新 Potential 的设计规则

添加新 potential 时：

1. 继承 `BasePotential`
2. 用向量化 NumPy 实现 `value(r)` 和 `force(r)`
3. 填充 `_params`、`_param_names`、`_dparam_names`、`_param_linear_mask`
4. 提供 `_df_dparam_names` 或覆盖 `basis_values(r)` 以高效支持 FM
5. 如果 REM/CDREM Hessian 支持重要，提供 `_d2param_names`
6. 若存在自然 bounds，提供 `param_bounds()`
7. 若是公开 style 集合的一部分，在 `POTENTIAL_REGISTRY` 注册

避免：

- 在 potential 内嵌入原子 index 或拓扑逻辑
- 每个样本分配 Python 对象
- 在生产路径中依赖有限差分 `force_grad()`
- 在 potential 内部读取 workflow 或 optimizer 状态

---

## Workflow 合法访问模式

**Do**：

- 通过 `Forcefield` 组装或序列化辅助函数构建 potential
- 将 potential 作为低层科学模型对象对待
- 让 compute / trainer / solver 代码通过 `Forcefield` 消耗它们

**Do not**：

- 在 workflow 代码中直接调用 potential 方法来复现 compute 数学
- 在 potential 使用中混入拓扑 index 或 MPI 逻辑
- 在拥有的 `Forcefield` / trainer / solver 之外 mutate potential 参数
