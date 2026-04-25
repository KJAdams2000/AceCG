# 05 Solver Module Developer Reference

*Updated: 2026-04-23.*

Solvers sit at the same layer as trainers. They consume accumulated statistics and return a parameter update or closed-form solution. They do not own MPI execution, frame extraction, or cache lifetime.

Current solver interface:

| File | Responsibility |
|---|---|
| `solvers/base.py` | `BaseSolver`, shared one-shot solve interface |
| `solvers/fm_matrix.py` | `FMMatrixSolver`, closed-form solve from standard FM statistics |

---

## Layer Boundaries

```text
topology + compute
  -> Forcefield and reducer output from run_post pickle files

solver
  -> consumes one batch, solves, returns dict

workflow / trainer
  -> decides when to call solver and where to propagate the result
```

A solver should own:

- solve configuration
- a private `Forcefield` copy
- matrix algebra

A solver should not own:

- trajectory I/O
- MPI engine creation
- frame-selection strategy
- runtime cache
- training-loop control

---

## `BaseSolver`

### Responsibilities

| Method | Meaning |
|---|---|
| `__init__(forcefield, logger=None)` | Deep-copy and own a `Forcefield` |
| `schema()` | Describe the input/output dict contract |
| `get_params()` | Return the current full parameter vector |
| `update_forcefield(params)` | Write the full parameter vector back to the owned `Forcefield` |
| `solve(batch)` | Abstract one-shot solve entry point |

Rules for new solvers:

- accept a canonical batch dict
- return a plain result dict
- fail fast on shape or contract mismatches
- keep workflow and training concerns outside solver code

---

## `FMMatrixSolver`

`FMMatrixSolver` is the canonical closed-form FM solver. It consumes the `step_mode="fm"` `run_post()` pickle output and solves in one call.

### Supported Modes

| Mode | Meaning |
|---|---|
| `ols` | Ordinary least squares |
| `ridge` | Tikhonov-regularized solve |
| `bayesian` | Diagonal ARD Bayesian evidence updates |

### Construction

```python
from AceCG.solvers.fm_matrix import FMMatrixSolver

solver = FMMatrixSolver(
    trainer.forcefield,
    mode="ridge",
    ridge_alpha=1.0e-6,
)
```

The solver owns an independent `Forcefield` copy. Updating solver parameters does not mutate the caller's forcefield until the workflow chooses to propagate the result.

---

## FM Solver Batch Contract

`FMMatrixSolver.solve(batch)` expects the standard FM payload from a `run_post` pickle:

| Key | Shape | Meaning |
|---|---:|---|
| `JtJ` | `(p, p)` | Normalized FM normal matrix |
| `Jty` | `(p,)` | Normalized FM right-hand-side vector |
| `y_sumsq` | scalar | Normalized target-force squared norm |
| `nframe` | scalar | Number of contributing frames |
| `weight_sum` | scalar | Total frame weight before normalization |
| `n_atoms_obs` | scalar | Number of observed atoms per frame |

Optional fields:

| Key | Meaning |
|---|---|
| `step_index` | Copied into `result["meta"]` when present |

### Result Contract

| Key | Meaning |
|---|---|
| `params` | Full solved parameter vector |
| `loss` | Normalized FM loss at solved parameters |
| `mode` | Solve mode |
| `meta` | Diagnostics such as active-parameter count and Bayesian statistics |

---

## Scientific Contract for FM Statistics

The FM reducer returns normalized quadratic statistics:

$$JtJ = \sum_i w_i J_i^T J_i,\quad Jty = \sum_i w_i J_i^T y_i,\quad y_\text{sumsq} = \sum_i w_i y_i^T y_i$$

where normalized weights satisfy $\sum_i w_i = 1$.

The solver uses:

$$L(\theta) = \frac{1}{2}\left(\theta^T JtJ\, \theta - 2\,\theta^T Jty + y_\text{sumsq}\right)$$

In Bayesian mode, `weight_sum` is used to reconstruct the unnormalized system:

$$JtJ_{\text{raw}} = \text{weight\_sum} \cdot JtJ,\quad Jty_{\text{raw}} = \text{weight\_sum} \cdot Jty$$

Effective scalar observation count:

$$N_{\text{obs}} = 3 \cdot n_\text{atoms\_obs} \cdot \text{weight\_sum}$$

This is why `weight_sum` and `n_atoms_obs` must remain in the FM payload.

---

## Mask Semantics

This is the most important developer contract for solvers.

### Rule 1: Compute Full FM Statistics

Even if a solver freezes some parameters, the compute path must still produce the full `JtJ` and `Jty`. Otherwise, cross terms between active and frozen parameters are lost.

In practice:

```python
ff_compute = Forcefield(trainer.forcefield)
ff_compute.param_mask = np.ones(ff_compute.n_params(), dtype=bool)
# use ff_compute as the forcefield in the run_post spec
```

### Rule 2: Apply Active Mask Inside the Solver

`FMMatrixSolver` reads the active mask from its owned `Forcefield.param_mask`.

If active indices are `a` and frozen indices are `f`, it solves the shifted system:

$$JtJ_{aa}\,\theta_a = Jty_a - JtJ_{af}\,\theta_f$$

This correctly preserves nonzero frozen parameters.

### Rule 3: Solver Mode Does Not Depend on Optimizer Objects

The workflow may mirror `trainer.optimizer.mask` into `solver.forcefield.param_mask`, but the solver itself depends only on `Forcefield.param_mask`.

---

## Solve Paths

### OLS

Performs diagonally scaled least-squares on the active block.

### Ridge

Solves:

$$\left(JtJ_{aa} + \lambda I\right)\theta_a = Jty_a^*$$

where $Jty_a^*$ is the shifted right-hand side described above.

### Bayesian

Performs diagonal ARD updates on the active block:

- reconstructs raw statistics from `weight_sum`
- uses Cholesky factorization and triangular solves, not dense matrix inverse
- reports convergence and posterior hyperparameters in `meta`

---

## Minimal Workflow Call Pattern

```python
import pickle

with open(spec_step["output_file"], "rb") as f:
    stats = pickle.load(f)

ff_solve = copy.deepcopy(trainer.forcefield)
solver = FMMatrixSolver(ff_solve, mode="ridge", ridge_alpha=alpha)

result = solver.solve(stats)

trainer.update_forcefield(result["params"])
```
