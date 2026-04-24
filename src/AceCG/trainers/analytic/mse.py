# AceCG/trainers/analytic/mse.py
"""Analytic MSE (PMF-matching) trainer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from typing import TypedDict, NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import TypedDict, NotRequired

from ..base import BaseTrainer
from .rem import EnsembleBatch


# -----------------------------------------------------------------------------
# TypedDict schemas
# -----------------------------------------------------------------------------

class MSEBatch(TypedDict, total=False):
    """
    Batch schema for MSETrainerAnalytic.step.

    Required keys
    -------------
    pmf_AA : np.ndarray, shape (n_bins,)
    pmf_CG : np.ndarray, shape (n_bins,)
    CG_bin_idx_frame : np.ndarray, shape (n_frames,)
        Bin index per frame (0..n_bins-1).

    CG-side energy gradients (exactly one of the following)
    -------------------------------------------------------
    engine_stats : dict
        Pre-computed CG statistics from the one-pass engine.
        Must contain ``energy_grad_frame`` : (n_frames, n_params) and
        ``energy_grad_avg`` : (n_params,).
    CG : EnsembleBatch  (deprecated — requires pair2distance_frame path)

    Optional keys
    -------------
    weighted_gauge : bool
    weighted_loss : bool
        If True, use the weighted PMF-mismatch objective
            loss = Σ_s p_CG(s) [F_AA(s) - F_CG(s)]^2
        and the corresponding weighted gradient. If False, use the unweighted
        squared mismatch summed over bins.
    step_index : int
    """
    pmf_AA: Any
    pmf_CG: Any
    CG: NotRequired[EnsembleBatch]
    CG_bin_idx_frame: Any
    engine_stats: NotRequired[Any]
    weighted_gauge: NotRequired[bool]
    weighted_loss: NotRequired[bool]
    step_index: NotRequired[int]


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


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

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
        "engine_stats": (
            "required dict; pre-computed CG statistics; must contain "
            "'energy_grad_frame' : (n_frames, n_params) per-frame energy gradients and "
            "'energy_grad_avg' : (n_params,) weighted mean"
        ),
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
        CG_bin_idx_frame,
        *,
        engine_stats: Dict[str, Any],
        weighted_loss: bool = False,
        beta: float = 1.0,
        step_index: int = 0,
    ) -> MSEBatch:
        """
        Build a MSEBatch dict for MSETrainerAnalytic.step().

        Parameters
        ----------
        pmf_AA, pmf_CG
            PMFs defined on the same bins; shape (n_bins,).
        CG_bin_idx_frame
            Bin index per frame; shape (n_frames,).
        engine_stats
            Pre-computed CG statistics from the one-pass engine.
            Must contain ``energy_grad_frame`` : (n_frames, n_params) and
            ``energy_grad_avg`` : (n_params,).
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

        Returns
        -------
        MSEBatch
        """
        batch: MSEBatch = {
            "pmf_AA": pmf_AA,
            "pmf_CG": pmf_CG,
            "engine_stats": engine_stats,
            "CG_bin_idx_frame": CG_bin_idx_frame,
            "weighted_loss": bool(weighted_loss),
            "beta": float(beta),
            "step_index": int(step_index),
        }
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
        CG_bin_idx_frame = batch["CG_bin_idx_frame"]
        weighted_loss = bool(batch.get("weighted_loss", False))
        beta = float(batch.get("beta", 1.0))
        step_index = int(batch.get("step_index", 0))

        # --- CG-side energy gradients from engine_stats ---
        engine_stats = batch.get("engine_stats")
        if engine_stats is None:
            raise KeyError(
                "batch['engine_stats'] is required.  Use make_batch() with "
                "pre-computed CG statistics from the one-pass engine."
            )
        energy_grad_CG_frame = np.asarray(engine_stats["energy_grad_frame"], dtype=np.float64)
        energy_grad_CG = np.asarray(engine_stats["energy_grad_avg"], dtype=np.float64)

        # --- Per-bin conditional averages (inlined) ---
        n_frames = energy_grad_CG_frame.shape[0]
        frame_weight = np.ones(n_frames, dtype=np.float64)
        frame_weight /= frame_weight.sum()
        bin_idx = np.asarray(CG_bin_idx_frame)
        idx_set = set(bin_idx.tolist()) if hasattr(bin_idx, "tolist") else set(bin_idx)
        energy_grad_CG_given_bin: Dict[int, np.ndarray] = {}
        for sidx in idx_set:
            mask = bin_idx == sidx
            w_bin = frame_weight[mask]
            w_bin = w_bin / w_bin.sum()
            energy_grad_CG_given_bin[sidx] = energy_grad_CG_frame[mask].T @ w_bin

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
        grad = np.zeros_like(energy_grad_CG, dtype=float)
        for sidx in idx_set:
            weight_bin = p_AA[sidx] if weighted_loss else 1.0
            grad += weight_bin * delta[sidx] * (energy_grad_CG_given_bin[sidx] - energy_grad_CG)

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
