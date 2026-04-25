# 06 Optimizer Module Developer Reference

*Updated: 2026-04-23.*

Optimizers are parameter-update engines used by trainers. They own update state and masked stepping. They do not own forcefields, compute runtime, frame statistics, or workflow control.

---

## Core Modules

| File | Responsibility |
|---|---|
| `optimizers/base.py` | `BaseOptimizer`, shared optimizer contract |
| `optimizers/adam.py` | Masked Adam optimizer |
| `optimizers/adamW.py` | AdamW with decoupled weight decay |
| `optimizers/rmsprop.py` | Mask-aware RMSprop optimizer |
| `optimizers/newton_raphson.py` | Masked Newton-Raphson optimizer |
| `optimizers/multithreaded/adam.py` | Special multithreaded Adam variant; not exported on the main path |

`AceCG.optimizers.__init__` currently re-exports only the main single-process optimizers.

---

## Layer Boundaries

```text
trainer
  -> computes grad / hessian and decides whether to call optimizer.step()

optimizer
  -> mutates its own parameter vector L and returns delta_L

forcefield / workflow
  -> remain outside the optimizer; trainers synchronize them after a step
```

An optimizer should own:

- the current flattened parameter vector `L`
- trainable mask `mask`
- update-state buffers, such as moments or Hessian history
- the update rule

An optimizer should not own:

- `Forcefield`
- parameter bounds
- MPI or reducer state
- logging policy
- batch construction

---

## Public Optimizer Interface

`AceCG.optimizers` exports:

| Symbol | Meaning |
|---|---|
| `BaseOptimizer` | Abstract optimizer base class |
| `AdamMaskedOptimizer` | First-order Adam |
| `AdamWMaskedOptimizer` | AdamW with decoupled weight decay |
| `RMSpropMaskedOptimizer` | Mask-aware RMSprop |
| `NewtonRaphsonOptimizer` | Second-order Newton step |

---

## `BaseOptimizer`

### State

| Field | Meaning |
|---|---|
| `L` | Current flattened parameter vector |
| `mask` | Boolean trainable mask with the same shape as `L` |
| `lr` | Learning rate / step scaling parameter |

### Methods

| Method | Meaning |
|---|---|
| `__init__(L, mask, lr)` | Initialize parameter vector, trainable mask, and learning rate |
| `set_params(L_new)` | Replace the internal parameter vector |
| `state_dict()` | Serialize optimizer state |
| `load_state_dict(state)` | Restore optimizer state |
| `step(grad, hessian=None)` | Abstract single-step update |

### Return-Value Convention

Optimizers mutate `self.L` in place and return the change in the full parameter space:

$$\Delta L = L_{\text{new}} - L_{\text{old}}$$

- masked-out coordinates return 0
- gradient-descent optimizers usually return negative deltas
- trainer code can log `update_norm` directly from the returned vector

---

## Mask Semantics

All current optimizers are mask-aware:

- only coordinates with `mask == True` are updated
- masked-out coordinates remain unchanged
- the returned delta vector always has full length

Example:

```python
opt = AdamMaskedOptimizer(L0, mask=np.array([True, False, True]), lr=1e-2)
delta = opt.step(grad)
# opt.L[1] is unchanged, delta[1] == 0
```

Important: `Forcefield.param_mask` and `Forcefield.key_mask` are the canonical source of trainability. Optimizers receive a mask copy at initialization for execution state; workflows should not treat `optimizer.mask` as an independent source of truth.

---

## Hessian Contract

Only some optimizers consume Hessians. Trainers discover this through `BaseTrainer.optimizer_accepts_hessian()`, which checks whether the optimizer's `step()` signature has a parameter named exactly `hessian`.

Developer rules:

- if an optimizer needs Hessians, its `step()` signature must contain `hessian`
- otherwise trainer code treats it as a first-order optimizer

Example:

```python
def step(self, grad: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    ...
```

---

## Optimizer Notes

### `AdamMaskedOptimizer`

Internal state:

- first moment `m`
- second moment `v`
- step counter `t`
- optional preconditioned Gaussian noise

Update rule: standard Adam on masked coordinates, returning `delta_L`.

### `AdamWMaskedOptimizer`

Adds to Adam:

- decoupled weight decay
- optional AMSGrad path
- optional preconditioned Gaussian noise

Use this when explicit decoupled weight decay is preferred over L2-through-gradient coupling.

### `RMSpropMaskedOptimizer`

Internal state:

- running mean of squared gradients
- optional momentum buffer
- optional centered-RMSprop state
- optional noise

Supports masked updates, optional L2-style weight decay through the gradient, and optional momentum.

### `NewtonRaphsonOptimizer`

Consumes both gradient and Hessian and computes a masked Newton step on the active block.

This is the current second-order path used by Hessian-capable analytic trainers.

Additional state for downstream logs:

- `last_grad`
- `last_hessian`
- `last_update`

### `optimizers/multithreaded/adam.py`

A special variant outside the main export path. Treat it as an implementation extension rather than the baseline optimizer contract.

---

## State Serialization

All optimizers should support:

```python
state = opt.state_dict()
opt.load_state_dict(state)
```

Base fields:

| Key | Meaning |
|---|---|
| `L` | Parameter vector |
| `mask` | Trainable mask |
| `lr` | Learning rate |

Subclasses add moment buffers, counters, and other internal state as needed.
