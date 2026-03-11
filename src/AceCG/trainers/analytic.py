# AceCG/trainers/analytic.py
"""Analytic trainers (non-NN) with dict-based I/O.

This module provides analytic (NumPy-based) training loops for coarse-grained potentials.

Key design choices
------------------
1) **Dictionary-based input**: `trainer.step(batch=...)` where `batch` is a dict.
   This makes call sites explicit and resilient to signature drift.

2) **Dictionary-based output**: every `step()` returns a dict with standardized keys:
   - "name": str
   - "grad": np.ndarray, shape (n_params,)
   - "hessian": np.ndarray | None, shape (n_params, n_params)
   - "update": np.ndarray, shape (n_params,)  (zeros if apply_update=False)
   - plus method-specific keys (e.g., "loss", "dUdL_AA", "dUdL_CG", "meta"...)

3) **Dry-run support**: `apply_update=False` computes gradients (and Hessians) but
   does not modify optimizer state nor parameters. This is required to implement
   true meta-optimization in `MultiTrainerAnalytic` (combine gradients then step once).

Expected batch schemas
----------------------
REMTrainerAnalytic.step(batch):
  batch = {
    "CG": {"dist": ..., "weight": optional},
    "recompute_dUdL_AA": optional bool,   # default True
    # If recompute_dUdL_AA=True:
    "AA": {"dist": ..., "weight": optional},
    # If recompute_dUdL_AA=False:
    "dUdL_AA": np.ndarray,                # precomputed <dU/dλ>_AA
    "d2U_AA": np.ndarray,                 # optional; required only when optimizer needs Hessian
    "step_index": optional int,
  }

MSETrainerAnalytic.step(batch):
  batch = {
    "pmf_AA": np.ndarray, shape (n_bins,),
    "pmf_CG": np.ndarray, shape (n_bins,),
    "CG": {"dist": ..., "weight": optional},
    "CG_bin_idx_frame": np.ndarray, shape (n_frames,),
    "weighted_gauge": optional bool,
    "weighted_loss": optional bool,
    "step_index": optional int,
  }

CDREMTrainerAnalytic.step(batch):
  batch = {
    "dUdL_z_by_x": np.ndarray, shape (n_x, n_params),
    "dUdL_xz": np.ndarray, shape (n_params,),
    "x_weight": optional np.ndarray, shape (n_x,),
    "step_index": optional int,
  }

MultiTrainerAnalytic.step(batches):
  batches = [batch_for_trainer0, batch_for_trainer1, ...]  (same length as trainers)
  Note: each sub-batch may include optional dUdL-parallel keys `parallel_dUdL`, `dUdL_n_parts`, `dUdL_n_workers` (forwarded to REM/MSE).
  For REM sub-batches, `recompute_dUdL_AA=False` allows reuse of precomputed AA derivatives.

"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# -----------------------------------------------------------------------------
# TypedDict schemas (IDE-friendly)
# -----------------------------------------------------------------------------
# These types are for developer ergonomics: IDE auto-completion and static checking.
# They do not change runtime behavior and remain compatible with plain dict inputs.

try:
    from typing import TypedDict, NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import TypedDict, NotRequired


class EnsembleBatch(TypedDict, total=False):
    """
    Per-ensemble data passed to analytic derivative routines.

    Keys
    ----
    dist : array-like (required by trainers using this batch)
        Per-frame geometric features (often pair distances). Consumed by
        `dUdLByFrame(potential, dist)`.
    weight : array-like, NotRequired
        Per-frame weights for reweighting ensemble averages; shape (n_frames,).
    """
    dist: Any
    weight: NotRequired[Any]


class REMBatch(TypedDict, total=False):
    """
    Batch schema for REMTrainerAnalytic.step.

    Required keys
    -------------
    CG : EnsembleBatch
        CG/model ensemble. Must include CG["dist"].

    AA-side inputs
    --------------
    Use exactly one of the following AA modes.

    Mode 1: recompute from AA["dist"] (default)
        recompute_dUdL_AA : bool = True
        AA : EnsembleBatch
            AA/reference ensemble. Must include AA["dist"].

    Mode 2: reuse precomputed AA derivatives
        recompute_dUdL_AA : bool = False
        dUdL_AA : np.ndarray
            Precomputed AA ensemble average <dU/dλ>_AA.
        d2U_AA : np.ndarray, optional
            Precomputed AA second-derivative matrix used in the REM Hessian.
            Required only when the optimizer accepts a Hessian.

    Optional keys
    -------------
    step_index : int
        Logging step counter.
    parallel_dUdL : bool
        Whether parallel dUdL calculation is enabled for recomputed paths.
    dUdL_n_parts : int
        Split Pair2DistanceByFrame into this many chunks for dUdL_parallel.
    dUdL_n_workers : int
        Number of worker processes for dUdL_parallel.
    """
    AA: NotRequired[EnsembleBatch]
    CG: EnsembleBatch
    recompute_dUdL_AA: NotRequired[bool]
    dUdL_AA: NotRequired[Any]
    d2U_AA: NotRequired[Any]
    step_index: NotRequired[int]
    parallel_dUdL: bool
    dUdL_n_parts: int
    dUdL_n_workers: Optional[int]


class REMOut(TypedDict, total=False):
    """
    Return schema for REMTrainerAnalytic.step.

    Common keys
    ----------
    name : str
    grad : np.ndarray
    hessian : np.ndarray | None
    update : np.ndarray
    meta : dict

    REM-specific keys
    -----------------
    dUdL_AA : np.ndarray
    dUdL_CG : np.ndarray
    """
    name: str
    grad: Any
    hessian: NotRequired[Any]
    update: Any
    dUdL_AA: Any
    dUdL_CG: Any
    meta: Dict[str, Any]


class MSEBatch(TypedDict, total=False):
    """
    Batch schema for MSETrainerAnalytic.step.

    Required keys
    -------------
    pmf_AA : np.ndarray, shape (n_bins,)
    pmf_CG : np.ndarray, shape (n_bins,)
    CG : EnsembleBatch
        Must include CG["dist"].
    CG_bin_idx_frame : np.ndarray, shape (n_frames,)
        Bin index per frame (0..n_bins-1).

    Optional keys
    -------------
    weighted_gauge : bool
    weighted_loss : bool
        If True, use the weighted PMF-mismatch objective
            loss = Σ_s p_CG(s) [F_AA(s) - F_CG(s)]^2
        and the corresponding weighted gradient. If False, use the unweighted
        squared mismatch summed over bins.
    step_index : int
    parallel_dUdL : bool
        Whether parallel dUdL calculation
    dUdL_n_parts : int
        Splitting Pair2Distance into n_parts
    dUdL_n_workers : int
        N_workers for dUdL parallelization
    """
    pmf_AA: Any
    pmf_CG: Any
    CG: EnsembleBatch
    CG_bin_idx_frame: Any
    weighted_gauge: NotRequired[bool]
    weighted_loss: NotRequired[bool]
    step_index: NotRequired[int]
    parallel_dUdL: bool
    dUdL_n_parts: int
    dUdL_n_workers: Optional[int]


class MSEOut(TypedDict, total=False):
    """
    Return schema for MSETrainerAnalytic.step.

    Common keys
    ----------
    name : str
    grad : np.ndarray
    hessian : None
    update : np.ndarray
    meta : dict

    MSE-specific keys
    -----------------
    loss : float
    """
    name: str
    loss: float
    grad: Any
    hessian: NotRequired[Any]
    update: Any
    meta: Dict[str, Any]


class CDREMBatch(TypedDict, total=False):
    """
    Batch schema for CDREMTrainerAnalytic.step.

    Required first-order keys
    -------------------------
    dUdL_z_by_x : np.ndarray, shape (n_x, n_params)
        Row i is the conditional first-derivative average for one x-subsample:
            E_{z~q(z|x_i)} [ dU/dλ ]
    dUdL_xz : np.ndarray, shape (n_params,)
        Joint-model first-derivative average:
            E_{(x,z)~q(x,z)} [ dU/dλ ]

    Optional weighting keys
    -----------------------
    x_weight : np.ndarray, shape (n_x,)
        Weight for each x-subsample. If omitted, uniform average over x is used.

    Optional second-order keys
    --------------------------
    d2U_z_by_x : np.ndarray, shape (n_x, n_params, n_params)
        Row i is the conditional second-derivative average for x_i:
            E_{z~q(z|x_i)} [ d²U/dλ_i dλ_j ]
    d2U_xz : np.ndarray, shape (n_params, n_params)
        Joint-model second-derivative average:
            E_{(x,z)~q(x,z)} [ d²U/dλ_i dλ_j ]
    dUdLdUdL_xz : np.ndarray, shape (n_params, n_params)
        Joint-model second moment of first derivatives:
            E_{(x,z)~q(x,z)} [ (dU/dλ)(dU/dλ)^T ]
    cov_z_by_x : np.ndarray, shape (n_x, n_params, n_params)
        Conditional covariance of first derivatives for each x_i:
            Cov_{z~q(z|x_i)} [ dU/dλ ]

    Misc
    ----
    step_index : int
        Logging step counter.
    """
    dUdL_z_by_x: Any
    dUdL_xz: Any
    x_weight: NotRequired[Any]
    d2U_z_by_x: NotRequired[Any]
    d2U_xz: NotRequired[Any]
    dUdLdUdL_xz: NotRequired[Any]
    cov_z_by_x: NotRequired[Any]
    step_index: NotRequired[int]


class CDREMOut(TypedDict, total=False):
    """
    Return schema for CDREMTrainerAnalytic.step.

    Common keys
    ----------
    name : str
    grad : np.ndarray
    hessian : np.ndarray | None
    update : np.ndarray
    meta : dict

    CDREM-specific keys
    -------------------
    dUdL_pos : np.ndarray
        Positive-phase derivative E_xE_{z|x}[dU/dλ].
    dUdL_neg : np.ndarray
        Negative-phase derivative E_{x,z}[dU/dλ].
    d2U_pos : np.ndarray, optional
        Positive-phase second-derivative average E_xE_{z|x}[d²U].
    d2U_neg : np.ndarray, optional
        Negative-phase second-derivative average E_{x,z}[d²U].
    cov_neg : np.ndarray, optional
        Joint covariance Cov_{x,z}[dU/dλ].
    cov_pos_cond : np.ndarray, optional
        Weighted conditional covariance E_x Cov_{z|x}[dU/dλ].
    """
    name: str
    grad: Any
    hessian: NotRequired[Any]
    update: Any
    dUdL_pos: Any
    dUdL_neg: Any
    d2U_pos: NotRequired[Any]
    d2U_neg: NotRequired[Any]
    cov_neg: NotRequired[Any]
    cov_pos_cond: NotRequired[Any]
    meta: Dict[str, Any]


class MultiOut(TypedDict, total=False):
    """Return schema for MultiTrainerAnalytic.step."""
    mode: str
    update: Any
    combined_grad: NotRequired[Any]
    combined_hessian: NotRequired[Any]
    sub: List[Dict[str, Any]]
    meta: Dict[str, Any]

from .base import BaseTrainer
from .utils import optimizer_accepts_hessian
from ..utils.compute import (
    dUdLByFrame,
    dUdL,
    d2UdLjdLk_Matrix,
    dUdLj_dUdLk_Matrix,
    Hessian,
    dUdLByBin, dUdL_parallel,

)


def _get_step_index(batch: Dict[str, Any]) -> int:
    """Helper: robustly extract step_index from a batch dict."""
    si = batch.get("step_index", 0)
    try:
        return int(si)
    except Exception:
        return 0


class REMTrainerAnalytic(BaseTrainer):
    """Analytic Relative Entropy Minimization (REM) trainer.

    Computes REM gradient:
        grad = β ( <dU/dλ>_AA - <dU/dλ>_CG )

    Optionally computes Hessian if the optimizer supports it.

    Returns a dict with standardized keys (see module docstring).
    """

    # ---- Public schema objects (for documentation / validation) ----
    # These describe the expected `batch` dict and returned `out` dict keys.
    BATCH_SCHEMA: Dict[str, Any] = {
        "CG": {
            "dist": "required; per-frame CG features passed to dUdLByFrame(potential, dist)",
            "weight": "optional; per-frame CG weights for reweighting; shape (n_frames,)",
        },
        "recompute_dUdL_AA": "optional bool; default True; if True, recompute AA derivatives from AA['dist']; if False, read precomputed AA derivatives from batch['dUdL_AA'] (and batch['d2U_AA'] when Hessian is needed)",
        "AA": {
            "dist": "required when recompute_dUdL_AA=True; per-frame AA features passed to dUdLByFrame(potential, dist)",
            "weight": "optional; per-frame AA weights for reweighting; shape (n_frames,)",
        },
        "dUdL_AA": "required when recompute_dUdL_AA=False; precomputed AA ensemble average <dU/dλ>_AA",
        "d2U_AA": "required when recompute_dUdL_AA=False and optimizer_accepts_hessian is True; precomputed AA second-derivative matrix used in Hessian construction",
        "step_index": "optional int; logging step counter (TensorBoard); default 0",
        "parallel_dUdL": "optional bool; default False; if True and dist is a dict (Pair2DistanceByFrame), compute dUdL in parallel via dUdL_parallel() for recomputed paths",
        "dUdL_n_parts": "optional int; default 8; number of frame chunks for dUdL_parallel()",
        "dUdL_n_workers": "optional int; default None; number of worker processes for dUdL_parallel()",

    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "REM"',
        "grad": "np.ndarray; shape (n_params,); beta*(<dU/dλ>_AA - <dU/dλ>_CG)",
        "hessian": "np.ndarray|None; shape (n_params,n_params); only if optimizer_accepts_hessian is True",
        "update": "np.ndarray; shape (n_params,); optimizer update if apply_update=True else zeros_like(grad)",
        "meta": "dict; diagnostics (step_index, grad_norm, update_norm, ...)",
        "dUdL_AA": "np.ndarray; shape (n_params,); AA ensemble average <dU/dλ>_AA",
        "dUdL_CG": "np.ndarray; shape (n_params,); CG ensemble average <dU/dλ>_CG",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}
    @staticmethod
    def make_batch(
        CG_dist,
        AA_dist=None,
        dUdL_AA=None,
        d2U_AA=None,
        AA_weight=None,
        CG_weight=None,
        recompute_dUdL_AA: bool = True,
        step_index: int = 0,
        parallel_dUdL: bool = False,
        dUdL_n_parts: int = 8,
        dUdL_n_workers: Optional[int] = None,
    ) -> REMBatch:
        """
        Build a REMBatch dict for REMTrainerAnalytic.step().

        This helper keeps call sites explicit and stable as the trainer evolves.

        Parameters
        ----------
        CG_dist
            Per-frame CG features. Passed to ``dUdLByFrame(potential, CG_dist)``
            (or ``dUdL_parallel`` if enabled).
        AA_dist
            Per-frame AA features. Required when ``recompute_dUdL_AA=True``.
            Passed to ``dUdLByFrame(potential, AA_dist)`` (or ``dUdL_parallel``
            if enabled).
        dUdL_AA
            Precomputed AA ensemble average ``<dU/dλ>_AA``. Required when
            ``recompute_dUdL_AA=False``.
        d2U_AA
            Precomputed AA second-derivative matrix used in the REM Hessian.
            Only needed when ``recompute_dUdL_AA=False`` and the optimizer
            requires a Hessian.
        AA_weight, CG_weight
            Optional per-frame weights for reweighting the AA/CG ensemble averages.
            Shape must match the canonical frame ordering used by ``dUdLByFrame``:
            ``sorted(dist.keys())`` when ``dist`` is a dict.
        recompute_dUdL_AA
            Controls AA-side derivative reuse.

            - ``True``: recompute ``dUdL_AA`` from ``AA_dist`` each step.
              If the optimizer accepts a Hessian, also recompute the AA
              second-derivative term from ``AA_dist``.
            - ``False``: read precomputed ``dUdL_AA`` from the batch.
              If the optimizer accepts a Hessian, ``d2U_AA`` must also be
              supplied in the batch.
        step_index
            Logging step counter (e.g., TensorBoard global step).

        Parallel dUdL options
        ---------------------
        parallel_dUdL
            If True, and ``AA_dist`` / ``CG_dist`` are dicts keyed by frame_id
            (Pair2DistanceByFrame-style), compute dUdL via multiprocessing using
            ``dUdL_parallel`` for recomputed paths.
        dUdL_n_parts
            Number of frame chunks (work units) used by ``dUdL_parallel``.
            A good default is ~ ``n_workers`` or ``2*n_workers``.
        dUdL_n_workers
            Number of worker processes used by ``dUdL_parallel``. If None, the
            executor chooses a default (typically CPU count).

        Returns
        -------
        REMBatch
            Batch dictionary in either recompute mode or cached-AA mode.

        Notes
        -----
        - The trainer will automatically fall back to the serial path if
          ``parallel_dUdL`` is True but the provided ``dist`` is not dict-like.
        - When providing weights with dict-like ``dist``, make sure weights are
          aligned to ``sorted(dist.keys())`` (or pass weights as a dict keyed by
          frame_id to avoid ordering issues).
        """
        batch: REMBatch = {
            "CG": {"dist": CG_dist},
            "recompute_dUdL_AA": bool(recompute_dUdL_AA),
            "step_index": int(step_index),
            "parallel_dUdL": bool(parallel_dUdL),
            "dUdL_n_parts": int(dUdL_n_parts),
        }

        if recompute_dUdL_AA:
            if AA_dist is None:
                raise ValueError("AA_dist is required when recompute_dUdL_AA=True.")
            batch["AA"] = {"dist": AA_dist}
            if AA_weight is not None:
                batch["AA"]["weight"] = AA_weight
        else:
            if dUdL_AA is None:
                raise ValueError("dUdL_AA is required when recompute_dUdL_AA=False.")
            batch["dUdL_AA"] = dUdL_AA
            if d2U_AA is not None:
                batch["d2U_AA"] = d2U_AA

        if CG_weight is not None:
            batch["CG"]["weight"] = CG_weight
        if dUdL_n_workers is not None:
            batch["dUdL_n_workers"] = int(dUdL_n_workers)
        return batch



    def step(self, batch: REMBatch, apply_update: bool = True) -> REMOut:
        """
        Execute one REM optimization step.

        Parameters
        ----------
        batch : REMBatch
            REM batch dictionary.

            AA-side data can be provided in two modes:

            1. Recompute mode (default)
               ``batch["recompute_dUdL_AA"] = True``
               Requires ``batch["AA"]["dist"]`` and optionally ``batch["AA"]["weight"]``.

            2. Cached-AA mode
               ``batch["recompute_dUdL_AA"] = False``
               Requires precomputed ``batch["dUdL_AA"]``.
               If the optimizer accepts a Hessian, also requires
               ``batch["d2U_AA"]``.

            CG-side data is always recomputed from ``batch["CG"]["dist"]``.

        apply_update : bool, default=True
            If True, apply the optimizer step and update the potential.
            If False, run in dry-run mode and return ``update=zeros_like(grad)``.

        Returns
        -------
        REMOut
            Dictionary with standardized REM outputs, including ``grad``,
            optional ``hessian``, and the AA/CG ensemble averages of ``dU/dλ``.

        Notes
        -----
        ``recompute_dUdL_AA`` controls AA-side derivative reuse. When the
        optimizer requires a Hessian, the same flag also controls whether the
        AA second-derivative term is recomputed from ``AA["dist"]`` or read
        from precomputed ``d2U_AA``.
        """
        # --- Parse batch ---
        assert isinstance(batch, dict), "REMTrainerAnalytic.step expects batch as a dict."
        CG = batch["CG"]
        step_index = _get_step_index(batch)
        recompute_dUdL_AA = bool(batch.get("recompute_dUdL_AA", True))
        need_hessian = optimizer_accepts_hessian(self.optimizer)

        w_CG = CG.get("weight", None)
        AA = batch.get("AA", None)
        w_AA = None if AA is None else AA.get("weight", None)

        # --- Optional parallel dUdL evaluation (dict/PBDist only) ---
        # We only enable multiprocessing when:
        #   (1) batch requests it, AND
        #   (2) dist is a dict keyed by frame_id (Pair2DistanceByFrame-style).
        parallel_dUdL = bool(batch.get("parallel_dUdL", False))
        n_parts = int(batch.get("dUdL_n_parts", 8))
        n_workers = batch.get("dUdL_n_workers", None)
        if n_workers is not None:
            n_workers = int(n_workers)

        CG_dist = CG["dist"]

        # --- AA-side derivatives ---
        if recompute_dUdL_AA:
            if AA is None or "dist" not in AA:
                raise KeyError("batch['AA']['dist'] is required when recompute_dUdL_AA=True.")
            AA_dist = AA["dist"]

            # AA gradient term: only needs <dU/dλ>_AA.
            if parallel_dUdL and isinstance(AA_dist, dict):
                dUdL_AA = dUdL_parallel(
                    self.potential,
                    AA_dist,
                    frame_weight=w_AA,
                    n_parts=n_parts,
                    n_workers=n_workers,
                    mode="avg",
                )
            else:
                dUdL_AA_frame = dUdLByFrame(self.potential, AA_dist)
                dUdL_AA = dUdL(dUdL_AA_frame, w_AA)

            if need_hessian:
                d2U_AA = d2UdLjdLk_Matrix(self.potential, AA_dist, w_AA)
            else:
                d2U_AA = None
        else:
            if "dUdL_AA" not in batch:
                raise KeyError("batch['dUdL_AA'] is required when recompute_dUdL_AA=False.")
            dUdL_AA = batch["dUdL_AA"]
            if need_hessian:
                if "d2U_AA" not in batch:
                    raise KeyError(
                        "batch['d2U_AA'] is required when recompute_dUdL_AA=False "
                        "and the optimizer requires a Hessian."
                    )
                d2U_AA = batch["d2U_AA"]
            else:
                d2U_AA = None

        # --- CG-side derivatives ---
        # If Hessian is needed, we must materialize dUdL_CG_frame (for <g_j g_k> term).
        if parallel_dUdL and isinstance(CG_dist, dict):
            if need_hessian:
                dUdL_CG, dUdL_CG_frame, frames_sorted = dUdL_parallel(
                    self.potential,
                    CG_dist,
                    frame_weight=w_CG,
                    n_parts=n_parts,
                    n_workers=n_workers,
                    mode="frame",
                )
            else:
                dUdL_CG = dUdL_parallel(
                    self.potential,
                    CG_dist,
                    frame_weight=w_CG,
                    n_parts=n_parts,
                    n_workers=n_workers,
                    mode="avg",
                )
                dUdL_CG_frame = None
        else:
            dUdL_CG_frame = dUdLByFrame(self.potential, CG_dist)
            dUdL_CG = dUdL(dUdL_CG_frame, w_CG)

        # --- REM gradient ---
        grad = self.beta * (dUdL_AA - dUdL_CG)

        # --- Optional Hessian ---
        hessian = None
        if need_hessian:
            d2U_CG = d2UdLjdLk_Matrix(self.potential, CG["dist"], w_CG)
            dUU_CG = dUdLj_dUdLk_Matrix(dUdL_CG_frame, w_CG)
            hessian = Hessian(self.beta, d2U_AA, d2U_CG, dUU_CG, dUdL_CG)

        # --- Optimization step (optional) ---
        if apply_update:
            if hessian is not None:
                update = self.optimizer.step(grad, hessian=hessian)
            else:
                update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            # Dry-run: do NOT touch optimizer state or parameters.
            update = np.zeros_like(grad)

        # --- Logging ---
        if self.logger is not None:
            mask_ratio = float(np.mean(self.optimizer.mask.astype(float)))
            self.logger.add_scalar("REM/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("REM/lr", float(getattr(self.optimizer, "lr", np.nan)), step_index)
            self.logger.add_scalar("REM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("REM/update_norm", float(np.linalg.norm(update)), step_index)
            if hessian is not None:
                self.logger.add_scalar("REM/hessian_cond", float(np.linalg.cond(hessian)), step_index)

        return {
            "name": "REM",
            "grad": grad,
            "hessian": hessian,
            "update": update,
            "dUdL_AA": dUdL_AA,
            "dUdL_CG": dUdL_CG,
            "meta": {
                "step_index": step_index,
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
                "recompute_dUdL_AA": recompute_dUdL_AA,
                "need_hessian": need_hessian,
            },
        }


class MSETrainerAnalytic(BaseTrainer):
    """Analytic PMF-matching trainer with a gauge-fixed mean-squared objective.

    Let

        ΔF(s) = F_CG(s) - c - F_AA(s)

    where ``c`` is a gauge constant used to remove the arbitrary additive offset
    of the PMF. This trainer chooses ``c`` consistently with the loss definition,
    so the gauge fixing and the objective always use the same weighting.

    Unweighted objective (default)
        ``loss = 1/2 Σ_s [F_CG(s) - c - F_AA(s)]^2``

        with gauge shift

        ``c = mean_s [F_CG(s) - F_AA(s)]``

    AA-weighted objective
        ``loss = 1/2 Σ_s p_AA(s) [F_CG(s) - c - F_AA(s)]^2``

        where ``p_AA(s)`` is reconstructed from the reference PMF via

        ``p_AA(s) ∝ exp(-beta * F_AA(s))``

        followed by normalization over bins, and the gauge shift is

        ``c = Σ_s p_AA(s) [F_CG(s) - F_AA(s)]``

    Using

        ``∂F_CG(s)/∂λ = ⟨dU/dλ⟩_{CG|s} - ⟨dU/dλ⟩_CG``

    the gradient is

    Unweighted gradient
        ``∂loss/∂λ = Σ_s ΔF(s) * (⟨dU/dλ⟩_{CG|s} - ⟨dU/dλ⟩_CG)``

    AA-weighted gradient
        ``∂loss/∂λ = Σ_s p_AA(s) ΔF(s) * (⟨dU/dλ⟩_{CG|s} - ⟨dU/dλ⟩_CG)``

    Returns a dict with standardized keys (see module docstring).
    """

    # ---- Public schema objects (for documentation / validation) ----
    BATCH_SCHEMA: Dict[str, Any] = {
        "pmf_AA": "required np.ndarray; shape (n_bins,); target/reference PMF",
        "pmf_CG": "required np.ndarray; shape (n_bins,); current CG PMF",
        "CG": {
            "dist": "required; per-frame CG features passed to dUdLByFrame(potential, dist)",
            "weight": "optional; per-frame CG weights for reweighting; shape (n_frames,)",
        },
        "CG_bin_idx_frame": "required np.ndarray or dict; integer bin index per frame (0..n_bins-1)",
        "weighted_loss": (
            "optional bool; default False; if False use the unweighted objective "
            "loss = 1/2 Σ_s [F_CG(s)-c-F_AA(s)]^2 with c = mean_s[F_CG(s)-F_AA(s)]; "
            "if True use the AA-weighted objective loss = 1/2 Σ_s p_AA(s)[F_CG(s)-c-F_AA(s)]^2, "
            "where p_AA(s) is reconstructed from pmf_AA as p_AA(s) ∝ exp(-beta * F_AA(s)) and "
            "c = Σ_s p_AA(s)[F_CG(s)-F_AA(s)]. The gradient uses the same weighting as the loss."
        ),
        "beta": "optional float; default 1.0; inverse-temperature factor used to reconstruct p_AA(s) ∝ exp(-beta * F_AA(s)) when weighted_loss=True",
        "step_index": "optional int; logging step counter; default 0",
        "parallel_dUdL": "optional bool; default False; if True and dist is a dict (Pair2DistanceByFrame), compute dUdL in parallel via dUdL_parallel()",
        "dUdL_n_parts": "optional int; default 8; number of frame chunks for dUdL_parallel()",
        "dUdL_n_workers": "optional int; default None; number of worker processes for dUdL_parallel()",

    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "MSE"',
        "loss": "float; gauge-fixed PMF mismatch objective; either 1/2 Σ_s [F_CG(s)-c-F_AA(s)]^2 or 1/2 Σ_s p_AA(s)[F_CG(s)-c-F_AA(s)]^2 when weighted_loss=True",
        "grad": "np.ndarray; shape (n_params,); gradient of the gauge-fixed PMF mismatch wrt λ, weighted consistently with the chosen loss",
        "hessian": "None; reserved for uniform interface",
        "update": "np.ndarray; shape (n_params,); optimizer update if apply_update=True else zeros_like(grad)",
        "meta": "dict; diagnostics (step_index, gauge_shift, weighted_loss, beta, grad_norm, update_norm, ...)",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}
    @staticmethod
    def make_batch(
        pmf_AA,
        pmf_CG,
        CG_dist,
        CG_bin_idx_frame,
        CG_weight=None,
        weighted_loss: bool = False,
        beta: float = 1.0,
        step_index: int = 0,
        parallel_dUdL: bool = False,
        dUdL_n_parts: int = 8,
        dUdL_n_workers: Optional[int] = None,
    ) -> MSEBatch:
        """
        Build a MSEBatch dict for MSETrainerAnalytic.step().

        Parameters
        ----------
        pmf_AA, pmf_CG
            PMFs defined on the same bins; shape (n_bins,).
        CG_dist
            Per-frame CG features (e.g., Pair2DistanceByFrame). Passed to
            ``dUdLByFrame(potential, CG_dist)`` (or ``dUdL_parallel`` if enabled).
        CG_bin_idx_frame
            Bin index per frame. Accepted formats:
              - np.ndarray of shape (n_frames,), aligned to the canonical frame ordering
                of ``CG_dist`` (``sorted(CG_dist.keys())`` when dict-like), OR
              - dict {frame_id: bin_idx} (recommended when CG_dist is dict-like)
        CG_weight
            Optional per-frame weights for CG reweighting. Same alignment rules as
            ``CG_bin_idx_frame``.
        weighted_loss
            Controls both the objective and the gauge shift so they remain
            mathematically consistent.

            - ``False``:
              ``loss = 1/2 Σ_s [F_CG(s)-c-F_AA(s)]^2``
              with ``c = mean_s[F_CG(s)-F_AA(s)]``.

            - ``True``:
              ``loss = 1/2 Σ_s p_AA(s)[F_CG(s)-c-F_AA(s)]^2``
              with ``p_AA(s) ∝ exp(-beta * F_AA(s))`` and
              ``c = Σ_s p_AA(s)[F_CG(s)-F_AA(s)]``.

              The gradient uses the same ``p_AA(s)`` weighting bin by bin.
        beta
            Inverse-temperature factor used to reconstruct
            ``p_AA(s) ∝ exp(-beta * F_AA(s))`` when ``weighted_loss=True``.
            Ignored when ``weighted_loss=False``.
        step_index
            Logging step counter.

        Parallel dUdL options
        ---------------------
        parallel_dUdL
            If True and ``CG_dist`` is dict-like, compute per-frame dUdL in parallel
            using multiprocessing via ``dUdL_parallel(mode="frame")``.
        dUdL_n_parts
            Number of frame chunks (work units) used by ``dUdL_parallel``.
            A good default is ~ ``dUdL_n_workers`` or ``2*dUdL_n_workers``.
        dUdL_n_workers
            Number of worker processes used by ``dUdL_parallel``. If None, the
            executor chooses a default (typically CPU count).

        Returns
        -------
        MSEBatch
            {
              "pmf_AA": pmf_AA,
              "pmf_CG": pmf_CG,
              "CG": {"dist": CG_dist, "weight": CG_weight?},
              "CG_bin_idx_frame": CG_bin_idx_frame,
              "weighted_loss": weighted_loss,
              "beta": beta,
              "step_index": step_index,
              "parallel_dUdL": parallel_dUdL,
              "dUdL_n_parts": dUdL_n_parts,
              "dUdL_n_workers": dUdL_n_workers,
            }

        Notes
        -----
        - MSE needs per-frame derivatives to compute per-bin conditional averages,
          so the trainer uses ``mode="frame"`` when parallelizing.
        - If you pass ``CG_bin_idx_frame`` as a dict, the trainer will align it to
          the returned ``frames_sorted`` automatically.
        """
        batch: MSEBatch = {
            "pmf_AA": pmf_AA,
            "pmf_CG": pmf_CG,
            "CG": {"dist": CG_dist},
            "CG_bin_idx_frame": CG_bin_idx_frame,
            "weighted_loss": bool(weighted_loss),
            "beta": float(beta),
            "step_index": int(step_index),
            "parallel_dUdL": bool(parallel_dUdL),
            "dUdL_n_parts": int(dUdL_n_parts),
        }
        if CG_weight is not None:
            batch["CG"]["weight"] = CG_weight
        if dUdL_n_workers is not None:
            batch["dUdL_n_workers"] = int(dUdL_n_workers)
        return batch



    def step(self, batch: MSEBatch, apply_update: bool = True) -> MSEOut:
        """
        Execute one PMF-matching MSE optimization step.

        Parameters
        ----------
        batch : MSEBatch
            MSE batch dictionary.

            ``weighted_loss`` determines both the loss and the gauge shift:

            - ``False``:
              ``loss = 1/2 Σ_s [F_CG(s)-c-F_AA(s)]^2``
              with ``c = mean_s[F_CG(s)-F_AA(s)]``.

            - ``True``:
              ``loss = 1/2 Σ_s p_AA(s)[F_CG(s)-c-F_AA(s)]^2``
              where ``p_AA(s) ∝ exp(-beta * F_AA(s))`` and is normalized over bins,
              with ``c = Σ_s p_AA(s)[F_CG(s)-F_AA(s)]``.

            In the weighted case, the gradient is weighted by the same ``p_AA(s)``
            factors bin by bin.

        apply_update : bool, default=True
            If True, apply the optimizer step and update the potential.
            If False, run in dry-run mode and return ``update=zeros_like(grad)``.

        Returns
        -------
        MSEOut
            Dictionary with standardized MSE outputs, including the scalar loss,
            gradient, and diagnostics describing whether the weighted or
            unweighted objective was used.
        """
        assert isinstance(batch, dict), "MSETrainerAnalytic.step expects batch as a dict."
        pmf_AA = np.asarray(batch["pmf_AA"], dtype=float)
        pmf_CG = np.asarray(batch["pmf_CG"], dtype=float)
        CG = batch["CG"]
        CG_bin_idx_frame = batch["CG_bin_idx_frame"]
        weighted_loss = bool(batch.get("weighted_loss", False))
        beta = float(batch.get("beta", 1.0))
        step_index = _get_step_index(batch)

        # --- Compute dU/dλ for CG (optionally in parallel) ---
        w_CG = CG.get("weight", None)

        parallel_dUdL = bool(batch.get("parallel_dUdL", False))
        n_parts = int(batch.get("dUdL_n_parts", 8))
        n_workers = batch.get("dUdL_n_workers", None)
        if n_workers is not None:
            n_workers = int(n_workers)

        CG_dist = CG["dist"]

        # MSE requires per-frame derivatives for dUdLByBin, so we need mode="frame".
        # If CG_dist is not dict-like, fall back to the original serial path.
        if parallel_dUdL and isinstance(CG_dist, dict):
            dUdL_CG, dUdL_CG_frame, frames_sorted = dUdL_parallel(
                self.potential,
                CG_dist,
                frame_weight=w_CG,
                n_parts=n_parts,
                n_workers=n_workers,
                mode="frame",
            )

            # IMPORTANT: CG_bin_idx_frame must be aligned to frames_sorted.
            # If the caller provides bin indices as a dict {frame_id: bin}, align here.
            if isinstance(CG_bin_idx_frame, dict):
                CG_bin_idx_frame = np.array([CG_bin_idx_frame[fr] for fr in frames_sorted], dtype=int)
        else:
            dUdL_CG_frame = dUdLByFrame(self.potential, CG_dist)
            dUdL_CG = dUdL(dUdL_CG_frame, w_CG)

        # --- Per-bin conditional averages ---
        _, _, dUdL_CG_given_bin = dUdLByBin(
            dUdL_CG_frame,
            CG_bin_idx_frame,
            w_CG,
        )

        # --- Reconstruct AA bin probabilities from PMF_AA when needed ---
        if weighted_loss:
            pmf_AA_shift = pmf_AA - np.min(pmf_AA)
            p_AA = np.exp(-beta * pmf_AA_shift)
            p_AA_sum = np.sum(p_AA)
            if p_AA_sum <= 0.0 or not np.isfinite(p_AA_sum):
                raise ValueError("Failed to reconstruct p_AA from pmf_AA; check beta and PMF values.")
            p_AA = p_AA / p_AA_sum
        else:
            p_AA = None

        # --- Gauge shift for PMF_CG (must match the loss weighting) ---
        if weighted_loss:
            c = float(np.sum((pmf_CG - pmf_AA) * p_AA))
        else:
            c = float(np.mean(pmf_CG - pmf_AA))
        pmf_CG_shifted = pmf_CG - c

        # --- Loss ---
        delta = pmf_CG_shifted - pmf_AA
        if weighted_loss:
            loss = float(0.5 * np.sum(p_AA * (delta ** 2)))
        else:
            loss = float(0.5 * np.sum(delta ** 2))

        # --- Gradient of loss w.r.t parameters ---
        grad = np.zeros_like(dUdL_CG, dtype=float)
        idx_set = set(CG_bin_idx_frame.tolist()) if hasattr(CG_bin_idx_frame, "tolist") else set(CG_bin_idx_frame)
        for idx in idx_set:
            weight_bin = p_AA[idx] if weighted_loss else 1.0
            grad += weight_bin * delta[idx] * (dUdL_CG_given_bin[idx] - dUdL_CG)

        # --- Optimization step (optional) ---
        if apply_update:
            update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            update = np.zeros_like(grad)

        # --- Logging ---
        if self.logger is not None:
            mask_ratio = float(np.mean(self.optimizer.mask.astype(float)))
            self.logger.add_scalar("MSE/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("MSE/lr", float(getattr(self.optimizer, "lr", np.nan)), step_index)
            self.logger.add_scalar("MSE/loss", loss, step_index)
            self.logger.add_scalar("MSE/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("MSE/update_norm", float(np.linalg.norm(update)), step_index)

        return {
            "name": "MSE",
            "loss": loss,
            "grad": grad,
            "hessian": None,  # keep key for uniformity
            "update": update,
            "meta": {
                "step_index": step_index,
                "gauge_shift": c,
                "weighted_loss": weighted_loss,
                "beta": beta,
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }


class CDREMTrainerAnalytic(BaseTrainer):
    """Analytic latent-variable / CDREM trainer.

    Implements the latent-variable CDREM gradient

        grad = β ( E_x E_{z|x}[dU/dλ] - E_{x,z}[dU/dλ] )

    and, when the optimizer requires a Hessian and the needed second-order
    statistics are provided in the batch, the latent-variable Hessian

        H = β ( E_xE_{z|x}[d²U] - E_{x,z}[d²U]
                + β ( Cov_{x,z}[dU/dλ] - E_x Cov_{z|x}[dU/dλ] ) )

    The second-order part is optional at the batch level. However, if the
    optimizer accepts a Hessian, the required second-order statistics must be
    provided; otherwise this trainer raises a ValueError.
    """

    BATCH_SCHEMA: Dict[str, Any] = {
        "dUdL_z_by_x": (
            "required np.ndarray; shape (n_x, n_params); row i is "
            "E_{z~q(z|x_i)}[dU/dλ]"
        ),
        "dUdL_xz": (
            "required np.ndarray; shape (n_params,); "
            "E_{(x,z)~q(x,z)}[dU/dλ]"
        ),
        "x_weight": (
            "optional np.ndarray; shape (n_x,); weight for each x-subsample; "
            "if omitted, uniform average over x is used"
        ),
        "d2U_z_by_x": (
            "optional np.ndarray; shape (n_x, n_params, n_params); row i is "
            "E_{z~q(z|x_i)}[d²U/dλ_i dλ_j]; required if optimizer needs Hessian"
        ),
        "d2U_xz": (
            "optional np.ndarray; shape (n_params, n_params); "
            "E_{(x,z)~q(x,z)}[d²U/dλ_i dλ_j]; required if optimizer needs Hessian"
        ),
        "dUdLdUdL_xz": (
            "optional np.ndarray; shape (n_params, n_params); "
            "E_{(x,z)~q(x,z)}[(dU/dλ)(dU/dλ)^T]; required if optimizer needs Hessian"
        ),
        "cov_z_by_x": (
            "optional np.ndarray; shape (n_x, n_params, n_params); row i is "
            "Cov_{z~q(z|x_i)}[dU/dλ]; required if optimizer needs Hessian"
        ),
        "step_index": "optional int; logging step counter; default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "CDREM"',
        "grad": (
            "np.ndarray; shape (n_params,); "
            "beta*(E_xE_{z|x}[dU/dλ] - E_{x,z}[dU/dλ])"
        ),
        "hessian": (
            "np.ndarray|None; shape (n_params,n_params); latent-variable Hessian "
            "if optimizer_accepts_hessian is True and second-order batch stats are provided"
        ),
        "update": (
            "np.ndarray; shape (n_params,); "
            "optimizer update if apply_update=True else zeros_like(grad)"
        ),
        "dUdL_pos": (
            "np.ndarray; shape (n_params,); positive-phase derivative "
            "E_xE_{z|x}[dU/dλ]"
        ),
        "dUdL_neg": (
            "np.ndarray; shape (n_params,); negative-phase derivative "
            "E_{x,z}[dU/dλ]"
        ),
        "d2U_pos": (
            "optional np.ndarray; shape (n_params,n_params); positive-phase "
            "second-derivative average E_xE_{z|x}[d²U]"
        ),
        "d2U_neg": (
            "optional np.ndarray; shape (n_params,n_params); negative-phase "
            "second-derivative average E_{x,z}[d²U]"
        ),
        "cov_neg": (
            "optional np.ndarray; shape (n_params,n_params); joint covariance "
            "Cov_{x,z}[dU/dλ]"
        ),
        "cov_pos_cond": (
            "optional np.ndarray; shape (n_params,n_params); weighted conditional "
            "covariance E_x Cov_{z|x}[dU/dλ]"
        ),
        "meta": "dict; diagnostics (step_index, grad_norm, update_norm, n_x, ...)",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}

    @staticmethod
    def make_batch(
        dUdL_z_by_x,
        dUdL_xz,
        x_weight=None,
        d2U_z_by_x=None,
        d2U_xz=None,
        dUdLdUdL_xz=None,
        cov_z_by_x=None,
        step_index: int = 0,
    ) -> CDREMBatch:
        """
        Build a CDREMBatch dict for CDREMTrainerAnalytic.step().

        Parameters
        ----------
        dUdL_z_by_x : np.ndarray, shape (n_x, n_params)
            Row i is the conditional first-derivative average for one x-subsample:
                E_{z~q(z|x_i)}[dU/dλ]
        dUdL_xz : np.ndarray, shape (n_params,)
            Joint-model first-derivative average:
                E_{(x,z)~q(x,z)}[dU/dλ]
        x_weight : np.ndarray, optional, shape (n_x,)
            Weight for each x-subsample. If None, uniform averaging over x is used.
        d2U_z_by_x : np.ndarray, optional, shape (n_x, n_params, n_params)
            Conditional second-derivative average for each x-subsample.
        d2U_xz : np.ndarray, optional, shape (n_params, n_params)
            Joint-model second-derivative average.
        dUdLdUdL_xz : np.ndarray, optional, shape (n_params, n_params)
            Joint-model second moment of first derivatives.
        cov_z_by_x : np.ndarray, optional, shape (n_x, n_params, n_params)
            Conditional covariance of first derivatives for each x-subsample.
        step_index : int
            Logging step counter.

        Returns
        -------
        CDREMBatch
            Batch dictionary for CDREMTrainerAnalytic.step().
        """
        batch: CDREMBatch = {
            "dUdL_z_by_x": dUdL_z_by_x,
            "dUdL_xz": dUdL_xz,
            "step_index": int(step_index),
        }
        if x_weight is not None:
            batch["x_weight"] = x_weight
        if d2U_z_by_x is not None:
            batch["d2U_z_by_x"] = d2U_z_by_x
        if d2U_xz is not None:
            batch["d2U_xz"] = d2U_xz
        if dUdLdUdL_xz is not None:
            batch["dUdLdUdL_xz"] = dUdLdUdL_xz
        if cov_z_by_x is not None:
            batch["cov_z_by_x"] = cov_z_by_x
        return batch

    def step(self, batch: CDREMBatch, apply_update: bool = True) -> CDREMOut:
        """
        Execute one CDREM optimization step.

        First-order statistics are always required. Second-order statistics are
        optional unless the optimizer accepts a Hessian, in which case they are
        required and this method raises a ValueError if any are missing.

        Parameters
        ----------
        batch : CDREMBatch
            CDREM batch dictionary.
        apply_update : bool, default=True
            If True, apply the optimizer step and update the potential.
            If False, run in dry-run mode and return update=zeros_like(grad).

        Returns
        -------
        CDREMOut
            Dictionary with standardized CDREM outputs.
        """
        assert isinstance(batch, dict), "CDREMTrainerAnalytic.step expects batch as a dict."

        dUdL_z_by_x = np.asarray(batch["dUdL_z_by_x"], dtype=float)
        dUdL_xz = np.asarray(batch["dUdL_xz"], dtype=float)
        step_index = _get_step_index(batch)
        need_hessian = optimizer_accepts_hessian(self.optimizer)

        if dUdL_z_by_x.ndim != 2:
            raise ValueError(
                f"batch['dUdL_z_by_x'] must be 2D, got shape {dUdL_z_by_x.shape}"
            )
        if dUdL_xz.ndim != 1:
            raise ValueError(
                f"batch['dUdL_xz'] must be 1D, got shape {dUdL_xz.shape}"
            )
        if dUdL_z_by_x.shape[1] != dUdL_xz.shape[0]:
            raise ValueError(
                "Dimension mismatch: dUdL_z_by_x has shape "
                f"{dUdL_z_by_x.shape}, but dUdL_xz has shape {dUdL_xz.shape}"
            )

        n_x = dUdL_z_by_x.shape[0]
        n_params = dUdL_z_by_x.shape[1]
        if n_x == 0:
            raise ValueError("batch['dUdL_z_by_x'] must contain at least one x-subsample.")

        x_weight = batch.get("x_weight", None)
        if x_weight is None:
            w_x = np.ones(n_x, dtype=float) / float(n_x)
        else:
            w_x = np.asarray(x_weight, dtype=float)
            if w_x.ndim != 1:
                raise ValueError(
                    f"batch['x_weight'] must be 1D, got shape {w_x.shape}"
                )
            if w_x.shape[0] != n_x:
                raise ValueError(
                    f"batch['x_weight'] length {w_x.shape[0]} != n_x {n_x}"
                )
            w_sum = float(np.sum(w_x))
            if w_sum == 0.0:
                raise ValueError("batch['x_weight'] sums to zero.")
            w_x = w_x / w_sum

        dUdL_pos = w_x @ dUdL_z_by_x
        dUdL_neg = dUdL_xz
        grad = self.beta * (dUdL_pos - dUdL_neg)

        d2U_pos = None
        d2U_neg = None
        cov_neg = None
        cov_pos_cond = None
        hessian = None

        if need_hessian:
            required = ["d2U_z_by_x", "d2U_xz", "dUdLdUdL_xz", "cov_z_by_x"]
            missing = [k for k in required if k not in batch]
            if missing:
                raise ValueError(
                    "CDREMTrainerAnalytic.step requires second-order batch statistics "
                    "when optimizer_accepts_hessian(self.optimizer) is True. Missing keys: "
                    + ", ".join(missing)
                )

            d2U_z_by_x = np.asarray(batch["d2U_z_by_x"], dtype=float)
            d2U_xz = np.asarray(batch["d2U_xz"], dtype=float)
            dUdLdUdL_xz = np.asarray(batch["dUdLdUdL_xz"], dtype=float)
            cov_z_by_x = np.asarray(batch["cov_z_by_x"], dtype=float)

            if d2U_z_by_x.ndim != 3 or d2U_z_by_x.shape != (n_x, n_params, n_params):
                raise ValueError(
                    "batch['d2U_z_by_x'] must have shape "
                    f"(n_x, n_params, n_params)=({n_x}, {n_params}, {n_params}), "
                    f"got {d2U_z_by_x.shape}"
                )
            if d2U_xz.ndim != 2 or d2U_xz.shape != (n_params, n_params):
                raise ValueError(
                    f"batch['d2U_xz'] must have shape ({n_params}, {n_params}), got {d2U_xz.shape}"
                )
            if dUdLdUdL_xz.ndim != 2 or dUdLdUdL_xz.shape != (n_params, n_params):
                raise ValueError(
                    "batch['dUdLdUdL_xz'] must have shape "
                    f"({n_params}, {n_params}), got {dUdLdUdL_xz.shape}"
                )
            if cov_z_by_x.ndim != 3 or cov_z_by_x.shape != (n_x, n_params, n_params):
                raise ValueError(
                    "batch['cov_z_by_x'] must have shape "
                    f"(n_x, n_params, n_params)=({n_x}, {n_params}, {n_params}), "
                    f"got {cov_z_by_x.shape}"
                )

            d2U_pos = np.tensordot(w_x, d2U_z_by_x, axes=(0, 0))
            d2U_neg = d2U_xz
            cov_neg = dUdLdUdL_xz - np.outer(dUdL_neg, dUdL_neg)
            cov_pos_cond = np.tensordot(w_x, cov_z_by_x, axes=(0, 0))

            hessian = self.beta * (
                d2U_pos - d2U_neg + self.beta * (cov_neg - cov_pos_cond)
            )

        if apply_update:
            if need_hessian:
                update = self.optimizer.step(grad, hessian=hessian)
            else:
                update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            update = np.zeros_like(grad)

        if self.logger is not None:
            mask_ratio = float(np.mean(self.optimizer.mask.astype(float)))
            self.logger.add_scalar("CDREM/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("CDREM/lr", float(getattr(self.optimizer, "lr", np.nan)), step_index)
            self.logger.add_scalar("CDREM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("CDREM/update_norm", float(np.linalg.norm(update)), step_index)
            self.logger.add_scalar("CDREM/n_x", float(n_x), step_index)

        out: CDREMOut = {
            "name": "CDREM",
            "grad": grad,
            "hessian": hessian,
            "update": update,
            "dUdL_pos": dUdL_pos,
            "dUdL_neg": dUdL_neg,
            "meta": {
                "step_index": step_index,
                "n_x": int(n_x),
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
                "used_hessian": bool(need_hessian),
            },
        }
        if d2U_pos is not None:
            out["d2U_pos"] = d2U_pos
        if d2U_neg is not None:
            out["d2U_neg"] = d2U_neg
        if cov_neg is not None:
            out["cov_neg"] = cov_neg
        if cov_pos_cond is not None:
            out["cov_pos_cond"] = cov_pos_cond
        return out


class MultiTrainerAnalytic(BaseTrainer):
    """Meta-trainer that combines multiple analytic trainers.

    This version is fully dict-based:
      - Inputs are per-trainer batch dicts.
      - Outputs are dicts with semantic keys (no positional index bookkeeping).

    Parameters
    ----------
    combine_mode : {"update", "grad"}
      - "update": run each sub-trainer normally and combine their returned "update".
      - "grad":   run each sub-trainer in dry-run mode (apply_update=False), combine
                  their returned "grad" (and "hessian" if available), then perform a
                  single meta optimizer step.

    Notes
    -----
    - In "grad" mode, only the meta optimizer state evolves. Sub-trainers are evaluated
      as pure gradient/Hessian providers.
    - After the meta update, all sub-trainers are synchronized to the meta parameter
      vector via `tr.update_potential(self.optimizer.L)`.
    """

    # ---- Public schema objects (for documentation / validation) ----
    STEP_SCHEMA: Dict[str, Any] = {
        "batches": "required Sequence[dict]; length == len(trainers); batches[i] must satisfy trainers[i].BATCH_SCHEMA. Each sub-batch may optionally include {parallel_dUdL, dUdL_n_parts, dUdL_n_workers} to enable frame-parallel dUdL inside REM/MSE trainers. For REM sub-batches, AA-side inputs may be supplied either as AA['dist'] with recompute_dUdL_AA=True, or as precomputed dUdL_AA (and d2U_AA when Hessian is required) with recompute_dUdL_AA=False. CDREM sub-batches accept dUdL_z_by_x, dUdL_xz, and optional x_weight.",
        "return_keys_list": "optional Sequence[Sequence[str]]; if provided, filters keys in out['sub'][i]",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "mode": 'str; "update" or "grad"',
        "update": "np.ndarray; shape (n_params,); meta update applied to optimizer.L",
        "sub": "list[dict]; sub-trainer outputs (full or filtered by return_keys_list)",
        "meta": "dict; diagnostics (update_norm, and grad_norm in grad mode, ...)",
        "combined_grad": "np.ndarray; shape (n_params,); only in mode=='grad'",
        "combined_hessian": "np.ndarray|None; shape (n_params,n_params); only in mode=='grad'",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `step` and `return` schema for introspection."""
        return {"step": cls.STEP_SCHEMA, "return": cls.RETURN_SCHEMA}


    def __init__(
        self,
        potential,
        optimizer,
        trainer_list: Sequence[BaseTrainer],
        weight_array: np.ndarray,
        beta: Optional[float] = None,
        logger=None,
        combine_mode: str = "update",
    ):
        super().__init__(potential, optimizer, beta, logger)

        assert isinstance(trainer_list, (list, tuple)) and len(trainer_list) > 0, (
            "trainer_list must be a non-empty list/tuple of trainers"
        )
        for i, tr in enumerate(trainer_list):
            assert isinstance(tr, BaseTrainer), f"trainer_list[{i}] is not a BaseTrainer"

        assert isinstance(weight_array, np.ndarray), "weight_array must be a NumPy array"
        assert weight_array.ndim == 1, "weight_array must be 1D"
        assert len(trainer_list) == weight_array.shape[0], "each trainer must have exactly one weight"

        assert combine_mode in ("update", "grad"), "combine_mode must be 'update' or 'grad'"
        self.combine_mode = combine_mode

        # Keep deep copy to avoid side-effects (preserve your original intent).
        self.trainers: List[BaseTrainer] = copy.deepcopy(list(trainer_list))
        self.weights = np.asarray(weight_array, dtype=float)

        # Optional: sanity check presence of meta-optimizer L
        assert hasattr(self.optimizer, "L"), "Meta-optimizer must expose attribute `.L`"
    @staticmethod
    def make_batches(
        *batches: Dict[str, Any],
        parallel_dUdL: Optional[bool] = None,
        dUdL_n_parts: Optional[int] = None,
        dUdL_n_workers: Optional[int] = None,
        recompute_dUdL_AA: Optional[bool] = None,
        override: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convenience helper to build the `batches` list for MultiTrainerAnalytic.step().

        This exists purely for ergonomics, so call sites read naturally and keep
        multi-objective wiring explicit.

        In addition, this helper can **inject common batch settings**
        into each sub-batch (REM/MSE), so you do not have to set these keys
        repeatedly at call sites. This includes frame-parallel dUdL settings
        and, for REM batches, the AA-side recomputation flag.

        Examples
        --------
        >>> rem_batch = REMTrainerAnalytic.make_batch(...)
        >>> mse_batch = MSETrainerAnalytic.make_batch(...)
        >>> batches = MultiTrainerAnalytic.make_batches(
        ...     rem_batch, mse_batch,
        ...     parallel_dUdL=True, dUdL_n_parts=16, dUdL_n_workers=16
        ... )
        >>> cached_rem_batch = REMTrainerAnalytic.make_batch(
        ...     CG_dist=CG_dist,
        ...     dUdL_AA=dUdL_AA_cached,
        ...     d2U_AA=d2U_AA_cached,
        ...     recompute_dUdL_AA=False,
        ... )
        >>> batches = MultiTrainerAnalytic.make_batches(
        ...     cached_rem_batch, mse_batch,
        ...     recompute_dUdL_AA=False
        ... )

        Parameters
        ----------
        *batches
            Per-trainer batch dicts. Ordering must match `self.trainers`.
        parallel_dUdL
            If not None, set `batch["parallel_dUdL"] = parallel_dUdL` for each
            sub-batch (unless `override=False` and the key already exists).
        dUdL_n_parts
            If not None, set `batch["dUdL_n_parts"] = dUdL_n_parts` for each sub-batch
            (unless `override=False` and the key already exists).
        dUdL_n_workers
            If not None, set `batch["dUdL_n_workers"] = dUdL_n_workers` for each sub-batch
            (unless `override=False` and the key already exists).
        recompute_dUdL_AA
            If not None, set `batch["recompute_dUdL_AA"] = recompute_dUdL_AA` for
            each sub-batch (unless `override=False` and the key already exists).
            This is mainly useful for REM batches. Non-REM trainers will simply
            ignore the extra key.
        override
            If True, overwrite existing keys in sub-batches. If False (default),
            preserve any per-batch custom settings already present.

        Returns
        -------
        list_of_batches : list[dict]
            The same objects passed in, collected into a list (mutated in-place
            if injection settings are provided).

        Notes
        -----
        - Frame-parallel dUdL is implemented inside REM/MSE trainers via
          `dUdL_parallel(...)` and only activates when the corresponding `dist`
          is dict-like (Pair2DistanceByFrame).
        - For REM batches, `recompute_dUdL_AA=False` means the AA-side
          `<dU/dλ>_AA` is expected to be present directly in the batch, and
          `d2U_AA` must also be present if the optimizer later requests a Hessian.
        - MultiTrainerAnalytic itself does NOT compute dUdL; it only forwards
          batches to sub-trainers (or injects these optional keys here).
        """
        out: List[Dict[str, Any]] = list(batches)

        # No-op fast path: keep original behavior when user does not request injection.
        if (
            (parallel_dUdL is None)
            and (dUdL_n_parts is None)
            and (dUdL_n_workers is None)
            and (recompute_dUdL_AA is None)
        ):
            return out

        for b in out:
            if not isinstance(b, dict):
                continue

            if parallel_dUdL is not None and (override or ("parallel_dUdL" not in b)):
                b["parallel_dUdL"] = bool(parallel_dUdL)

            if dUdL_n_parts is not None and (override or ("dUdL_n_parts" not in b)):
                b["dUdL_n_parts"] = int(dUdL_n_parts)

            if dUdL_n_workers is not None and (override or ("dUdL_n_workers" not in b)):
                b["dUdL_n_workers"] = int(dUdL_n_workers)

            if recompute_dUdL_AA is not None and (override or ("recompute_dUdL_AA" not in b)):
                b["recompute_dUdL_AA"] = bool(recompute_dUdL_AA)

        return out


    def step(
        self,
        batches: Sequence[Dict[str, Any]],
        return_keys_list: Optional[Sequence[Sequence[str]]] = None,
        parallel_grad: bool = False,
        n_workers: Optional[int] = None,
    ) -> MultiOut:
        """
        Perform one optimization step for a MultiTrainerAnalytic instance.

        This method coordinates multiple sub-trainers (e.g. REMTrainerAnalytic,
        MSETrainerAnalytic) and combines their contributions according to the
        configured ``combine_mode``.

        Two combine modes are supported:

        - ``combine_mode == "update"``:
            Each sub-trainer performs a full ``step`` with ``apply_update=True``.
            The resulting parameter updates are linearly combined and applied by
            the meta-optimizer. This mode updates sub-trainer optimizers internally
            and is therefore executed serially.

        - ``combine_mode == "grad"``:
            Each sub-trainer performs a dry-run step with ``apply_update=False``,
            returning gradients (and optionally Hessians) without modifying any
            optimizer or potential state. The gradients are combined and a single
            meta-optimizer step is applied. In this mode, sub-trainer evaluations
            can be executed in parallel.

        Parameters
        ----------
        batches : Sequence[Dict[str, Any]]
            A sequence of batch dictionaries, one per sub-trainer. Each batch must
            contain all keys required by the corresponding trainer's ``step`` method.
            The ordering of ``batches`` must match the ordering of ``self.trainers``.

            In particular, REM sub-batches now support two AA-side input modes:

            - recompute mode:
              ``{"AA": {"dist": ...}, "CG": {"dist": ...}, "recompute_dUdL_AA": True}``
            - cached-AA mode:
              ``{"dUdL_AA": ..., "CG": {"dist": ...}, "recompute_dUdL_AA": False}``

            If the active optimizer accepts a Hessian, cached-AA REM batches must
            also include ``d2U_AA``.

        return_keys_list : Optional[Sequence[Sequence[str]]], optional
            If provided, specifies which keys from each sub-trainer's output
            dictionary should be included in the returned ``sub_view`` field.
            The outer sequence must have the same length as ``self.trainers``.
            If ``None``, the full output dictionary of each sub-trainer is returned.

        parallel_grad : bool, default False
            If True and ``combine_mode == "grad"``, evaluate sub-trainer gradients
            in parallel using a thread pool. This only affects the dry-run gradient
            evaluation stage; the combination of gradients and the meta-optimizer
            update are always performed serially. This option has no effect when
            ``combine_mode == "update"``.

        n_workers : Optional[int], optional
            Number of worker threads used for parallel gradient evaluation when
            ``parallel_grad`` is True. If ``None``, a reasonable default based on
            the number of sub-trainers is used. This parameter is ignored when
            ``parallel_grad`` is False or when ``combine_mode == "update"``.

        Returns
        -------
        MultiOut
            A dictionary with the following keys:

            - ``"grad"`` :
                The combined gradient with respect to the global parameter vector
                ``L`` after applying trainer weights.

            - ``"hessian"`` :
                The combined Hessian matrix, if enabled and available; otherwise
                ``None``.

            - ``"update"`` :
                The parameter update applied by the meta-optimizer.

            - ``"sub_full"`` :
                A list of full output dictionaries returned by each sub-trainer
                ``step`` call.

            - ``"sub_view"`` :
                A list of filtered sub-trainer outputs containing only the keys
                specified by ``return_keys_list`` (or the full outputs if
                ``return_keys_list`` is ``None``).

        Notes
        -----
        - Sub-trainers may internally use multiprocessing for frame-parallel dUdL evaluation (keys: parallel_dUdL / dUdL_n_parts / dUdL_n_workers). This is independent of `parallel_grad`, which parallelizes across trainers via threads.
        - Parallel execution is implemented using threads rather than processes in
        order to avoid copying large batch data (e.g. distance arrays) between
        processes and to preserve trainer and potential state in the main thread.
        - Only the gradient evaluation stage is parallelized; all state-modifying
        operations are executed serially to ensure deterministic behavior.
    """
        assert isinstance(batches, (list, tuple)) and len(batches) == len(self.trainers), (
            "batches length must match the number of trainers"
        )
        if return_keys_list is not None:
            assert isinstance(return_keys_list, (list, tuple)) and len(return_keys_list) == len(self.trainers), (
                "return_keys_list length must match the number of trainers"
            )

        use_hessian = optimizer_accepts_hessian(self.optimizer)

        sub_full: List[Dict[str, Any]] = [] # full list of subtrainer output
        sub_view: List[Dict[str, Any]] = [] # return list of subtrainer output

        # ----------------------------
        # Mode A: combine sub-updates
        # ----------------------------
        if self.combine_mode == "update":
            updates = []

            for i, tr in enumerate(self.trainers):
                out_i = tr.step(batches[i], apply_update=True)
                assert isinstance(out_i, dict), "Sub-trainer step() must return a dict."
                assert "update" in out_i, "Sub-trainer output must include key 'update'."

                upd = np.asarray(out_i["update"])
                assert upd.shape == np.asarray(self.optimizer.L).shape, (
                    f"update shape mismatch for trainer {i}: {upd.shape} vs {np.asarray(self.optimizer.L).shape}"
                )
                updates.append(np.copy(upd))
                sub_full.append(out_i)

                if return_keys_list is None:
                    sub_view.append(out_i)
                else:
                    keys = return_keys_list[i]
                    sub_view.append({k: out_i.get(k, None) for k in keys})

            U = np.stack(updates, axis=0)         # (n_trainers, n_params)
            final_update = self.weights @ U       # (n_params,), linear combination of updates from subtrainers

            self.optimizer.L += final_update
            self.clamp_and_update()

            # Sync sub-trainers to the new global L
            for tr in self.trainers:
                tr.update_potential(self.optimizer.L)

            if self.logger is not None:
                self.logger.add_scalar("Multi/update_norm", float(np.linalg.norm(final_update)), _get_step_index(batches[0]))

            return {
                "mode": "update",
                "update": final_update,
                "sub": sub_view,
                "meta": {
                    "update_norm": float(np.linalg.norm(final_update)),
                },
            }

        # -----------------------------------------
        # Mode B: combine grads (+ Hessians) then step once (support multithreading)
        # -----------------------------------------
        grads = []
        Hs = []
        
        def _eval_one(args): # dry-run a trainer.step
            i, tr, b = args
            out_i = tr.step(b, apply_update=False)
            return i, out_i

        # 1) parallel and get out_i
        if parallel_grad and (n_workers is None or n_workers != 1):
            max_workers = n_workers or min(32, len(self.trainers))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_eval_one, [(i, tr, batches[i]) for i, tr in enumerate(self.trainers)]))
            # sorted by i, deterministic
            results.sort(key=lambda x: x[0])
        else:
            results = [(i, self.trainers[i].step(batches[i], apply_update=False)) for i in range(len(self.trainers))]

        # 2) fill out_i into grads/Hs/sub_view
        grads = []
        Hs = []
        sub_full = []
        sub_view = []

        for i, out_i in results:
            assert isinstance(out_i, dict), "Sub-trainer step() must return a dict."
            assert "grad" in out_i, "Sub-trainer output must include key 'grad'."

            gi = np.asarray(out_i["grad"])
            assert gi.shape == np.asarray(self.optimizer.L).shape, (
                f"grad shape mismatch for trainer {i}: {gi.shape} vs {np.asarray(self.optimizer.L).shape}"
            )
            grads.append(np.copy(gi))

            if use_hessian:
                Hi = out_i.get("hessian", None)
                if Hi is None:
                    Hs.append(None)
                else:
                    Hi = np.asarray(Hi)
                    assert Hi.shape == (gi.size, gi.size), (
                        f"hessian shape mismatch for trainer {i}: {Hi.shape} vs ({gi.size}, {gi.size})"
                    )
                    Hs.append(np.copy(Hi))
            else:
                Hs.append(None)

            sub_full.append(out_i)
            if return_keys_list is None:
                sub_view.append(out_i)
            else:
                keys = return_keys_list[i]
                sub_view.append({k: out_i.get(k, None) for k in keys})


        G = np.stack(grads, axis=0)               # (n_trainers, n_params)
        g_total = self.weights @ G                # (n_params,)

        H_total = None
        if use_hessian and all(h is not None for h in Hs):
            H_stack = np.stack(Hs, axis=0)        # (n_trainers, n_params, n_params)
            H_total = np.tensordot(self.weights, H_stack, axes=(0, 0))

        if H_total is not None:
            update = self.optimizer.step(g_total, hessian=H_total)
        else:
            update = self.optimizer.step(g_total)

        self.optimizer.L += update
        self.clamp_and_update()

        for tr in self.trainers:
            tr.update_potential(self.optimizer.L)

        # Logging
        step_index = _get_step_index(batches[0])
        if self.logger is not None:
            self.logger.add_scalar("Multi/grad_norm", float(np.linalg.norm(g_total)), step_index)
            self.logger.add_scalar("Multi/update_norm", float(np.linalg.norm(update)), step_index)
            if H_total is not None:
                try:
                    self.logger.add_scalar("Multi/hessian_cond", float(np.linalg.cond(H_total)), step_index)
                except Exception:
                    pass

        return {
            "mode": "grad",
            "combined_grad": g_total,
            "combined_hessian": H_total,
            "update": update,
            "sub": sub_view,
            "meta": {
                "grad_norm": float(np.linalg.norm(g_total)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }

    def set_lrs(self, lrs: Sequence[float]) -> None:
        """Set per-trainer learning rates."""
        assert isinstance(lrs, (list, tuple, np.ndarray)) and len(lrs) == len(self.trainers), (
            "lrs length must match the number of trainers"
        )
        for i, tr in enumerate(self.trainers):
            tr.optimizer.lr = float(lrs[i])