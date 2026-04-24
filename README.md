# AceCG

AceCG is a coarse-graining force-field training engine for workflows such as
FM, REM, CDREM, CDFM, and VP growth.

The active package lives in `src/AceCG/`. Config templates live in `configs/`.
Tracked experiment records and generated outputs live in `experiments/`.

Start with:

- `current.md` for the live repo state
- `docs/architecture.md` for code and folder boundaries
- `docs/workflow.md` for the config-to-experiment flow
- `docs/hpc.md` for Midway3 cluster rules

Typical lightweight check from the repo root:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Use a compute node for MPI, LAMMPS, production runs, and long test suites.

