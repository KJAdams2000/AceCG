# 10 VP Grower Developer Reference

*Updated: 2026-04-23.*

> This chapter covers only the VP grower pipeline. Training workflows are documented in [09_workflows.md](09_workflows.md).

In the active repo, VP grower is an independent one-shot data-production pipeline:

- it does not inherit from `BaseWorkflow`
- it does not participate in trainer / optimizer / checkpoint training loops
- it does not reuse `TaskScheduler.run_iteration()`

Its goal is to generate, from a CG-only reference topology and trajectory:

- a VP-augmented schema topology
- `latent.settings` and initial pair/bond/angle tables
- per-conditioning-frame `frame_*.data`
- optional `frame_*.forces.npy`
- `manifest.json`

---

## Core Files

| File | Responsibility |
|---|---|
| `workflows/vp_growth.py` | Top-level one-shot orchestrator; decides Universe loading strategy and output layout |
| `topology/vpgrower.py` | VP template construction, single-frame VP placement, `.data` writing |
| `compute/vp_prepare.py` | MPI-parallel frame growth and manifest aggregation |
| `io/vp_ffbuilder.py` | Export of `latent.settings` and initial VP tables |
| `configs/vp_growth_config.py` | VP grower-specific config model and parser |

---

## Static Template vs Dynamic Frames

The core design separates the static topology template from per-frame geometry.

| Layer | Main object | Description |
|---|---|---|
| Static template | `VPTopologyTemplate` | Atom names, type ids, inserted bonds / angles / dihedrals, real/VP index mappings |
| Single-frame geometry | `VPGrownFrame` | `(n_atoms, 3)` coordinates aligned to the template plus box dimensions |
| Executor | `VPGrower` | Holds the template and calls `grow_frame(...)` per frame |

This keeps shared data small across ranks: broadcast the template once instead of rebuilding topology every frame.

---

## Key Entry Points

| Symbol | Location | Purpose |
|---|---|---|
| `VPGrowthWorkflow.run()` | `workflows/vp_growth.py` | Top-level execution: read config, choose Universe strategy, write schema, dispatch frame growth |
| `VPGrower.from_universe()` | `topology/vpgrower.py` | Build a `VPTopologyTemplate` from a CG-only `MDAnalysis.Universe` |
| `VPGrower.grow_frame()` | `topology/vpgrower.py` | Insert VP coordinates into one frame of real-site coordinates |
| `grow_vp_frames()` | `compute/vp_prepare.py` | Split work by frame id and write `frame_*.data` in MPI parallel |
| `write_latent_settings()` | `io/vp_ffbuilder.py` | Export `latent.settings` and initial VP tables |
| `write_vp_data()` | `topology/vpgrower.py` | Write template plus single-frame geometry as LAMMPS data |

---

## `VPGrowthWorkflow`

`VPGrowthWorkflow` is a one-shot driver, not a training workflow.

Top-level steps:

1. Read `VPGrowthConfig`.
2. Decide whether to broadcast the full Universe or have each rank load only local segments.
3. Rank 0 builds `VPGrower` / `VPTopologyTemplate`.
4. Rank 0 writes `vp_topology.data` and `latent.settings`.
5. All ranks call `grow_vp_frames()`.
6. Rank 0 aggregates `manifest.json`.

The current trajectory-loading strategy is explicit:

- when the segment count is small, broadcast the full Universe
- when there are many segments, each rank opens only the trajectory subset it owns

This strategy lives directly in `workflows/vp_growth.py` and does not depend on the scheduler layer.

---

## `VPGrower`

`VPGrower` is a stateless-after-construction executor. It owns an immutable template; the per-frame inputs are:

- real-site positions
- box dimensions
- orientation seed

`topology/vpgrower.py` also owns:

- parsed VP bond / angle / dihedral specs
- carrier-residue to VP-slot mapping
- anti-clash iterative placement logic
- `write_vp_data()`, a thin wrapper around `write_lammps_data()`

The design boundary is deliberate: geometric placement is controlled only by VP bonds / angles and anti-clash logic. VP dihedrals are written into the final topology and `latent.settings`, but they do not participate in the geometric grow itself.

---

## `grow_vp_frames()`

`compute/vp_prepare.py` performs parallel frame growth. Its frame-id sharding intentionally matches `MPIComputeEngine.run_post()` so VP grower and compute runtime share discrete-frame semantics.

Important inputs:

- `grower`: shared-template `VPGrower`
- `universe`: locally seekable `MDAnalysis.Universe` on each rank
- `frame_ids`: global frame-id list
- `local_frame_ids`: optional local Universe seek indices for this rank
- `orientation_seed_base`: base random-orientation seed per frame

Output convention:

- `frame_{fid:06d}.data`
- optional `frame_{fid:06d}.forces.npy`
- rank 0 merges a `VPGrowManifest`

---

## `write_latent_settings()`

`io/vp_ffbuilder.py` converts the VP template and VP config into static files consumable by LAMMPS:

- `latent.settings`
- initial pair/bond/angle tables

Important helpers:

| Function | Purpose |
|---|---|
| `build_vp_forcefield()` | Materialize a VP-only `Forcefield` from `VPConfig + VPTopologyTemplate` |
| `render_vp_latent_template()` | Render final `pair_coeff` / `bond_coeff` / `angle_coeff` / `dihedral_coeff` text |

These outputs belong to the VP grower pipeline and are therefore not documented in the generic I/O chapter.

---

## Output Files

A VP grow run usually produces:

```text
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

Where:

- `vp_topology.data` is a schema-only topology used by later CDREM / CDFM runs
- `frame_*.data` files are the actual per-frame grown configurations
- `manifest.json` is the canonical rank-0 aggregated index

---

## Development Rules

When extending VP grower, preserve these boundaries:

1. Keep static topology templates in `topology/vpgrower.py`; do not move them into training workflows.
2. Keep parallel frame growth in `compute/vp_prepare.py`, preserving the same frame-id sharding semantics as `run_post()`.
3. Treat `latent.settings` and VP table output as VP-specific I/O; do not fold them into the generic `io` developer document.
