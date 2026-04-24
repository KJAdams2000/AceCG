# AceCG/trainers/base.py
from abc import ABC, abstractmethod
import numpy as np
import copy
from typing import Any, List, Optional, Tuple, Dict, NamedTuple

from ..potentials.base import IteratePotentials
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from ..potentials.base import BasePotential
from ..optimizers.base import BaseOptimizer

class BaseTrainer(ABC):
    """
    Abstract base class for analytic trainers in AceCG.

    Parameters
    ----------
    forcefield : dict
        Mapping from key → BasePotential or List[BasePotential]. Will be deep-copied.
    optimizer : BaseOptimizer
        Optimizer instance. Will be deep-copied.
    beta : float, optional
        Inverse temperature β.
    logger : SummaryWriter or None, optional
        Optional logger.

    Attributes
    ----------
    forcefield : dict
        Deep copy of the input forcefield dictionary.
    optimizer : BaseOptimizer
        Deep copy of the input optimizer.
    """
    def __init__(self, 
                 forcefield: Forcefield, 
                 optimizer: BaseOptimizer, 
                 beta: Optional[float] = None, 
                 logger=None):
        self.forcefield = copy.deepcopy(forcefield)
        self.optimizer = copy.deepcopy(optimizer)
        self.beta = beta
        self.logger = logger

    def get_params(self) -> np.ndarray:
        """
        Concatenate and return the current parameter vector from all potentials.

        Returns
        -------
        np.ndarray
            1D parameter vector matching optimizer.L. Shape: (n_params,).
        """
        return self.forcefield.param_array()

    def update_forcefield(self, L_new: np.ndarray):
        """
        Update `self.forcefield` and `self.optimizer` with a new parameter vector.
        """
        self.forcefield.update_params(L_new)
        self.optimizer.set_params(L_new)

    def clamp_and_update(self):
        """
        Clamp `self.optimizer.L` to [lb, ub] (if set) and propagate back to potentials.

        Notes
        -----
        - Calls `self.update_forcefield(self.optimizer.L)` after clamping
          to keep potential objects in sync with the optimizer state.
        """
        if getattr(self.optimizer, "L", None) is None:
            return
        Lc = self.forcefield.apply_bounds(self.optimizer.L)
        if not np.shares_memory(Lc, self.optimizer.L) or not np.allclose(Lc, self.optimizer.L):
            self.optimizer.set_params(Lc)
        self.update_forcefield(self.optimizer.L)

    def get_param_names(self) -> List[str]:
        """Return ordered list of human-readable parameter names."""
        names = []
        for key, pot in IteratePotentials(self.forcefield):
            prefix = key.label()
            for i in range(pot.n_params()):
                names.append(f"{prefix}[{i}]")
        return names

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower_bounds, upper_bounds) arrays."""
        return self.forcefield.param_bounds

    def get_interaction_labels(self) -> List[str]:
        """Return ordered list of interaction labels."""
        return [key.label() for key in self.forcefield.keys()]

    def n_total_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.n_params() for _, p in IteratePotentials(self.forcefield))

    def active_interaction_mask(self) -> dict:
        """Return L1 interaction mask derived from the L2 parameter mask.

        Returns a ``{key: bool}`` dict.  A key is ``True`` when at least
        one of its parameters is trainable (active in the L2 mask).
        """
        n = self.n_total_params()
        mask = getattr(self.optimizer, "mask", None)
        l2 = np.asarray(mask, dtype=bool) if mask is not None else np.ones(n, dtype=bool)
        if l2.shape != (n,):
            raise ValueError(
                f"optimizer.mask shape mismatch: expected {(n,)}, got {l2.shape}"
            )
        return self.forcefield.derive_l1_mask(l2)

    def is_optimization_linear(self) -> bool:
        """Return True when all active optimization channels are linear."""
        n_params = self.forcefield.n_params()

        mask = getattr(self.optimizer, "mask", None)
        active_mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones(n_params, dtype=bool)
        if active_mask.shape != (n_params,):
            raise ValueError(
                f"Active parameter mask shape mismatch: expected {(n_params,)}, got {active_mask.shape}"
            )

        offset = 0
        for _, pot in IteratePotentials(self.forcefield):
            linear_mask = np.asarray(pot.is_param_linear(), dtype=bool)
            n_local = pot.n_params()
            if linear_mask.shape != (n_local,):
                raise ValueError(
                    f"Linear mask shape mismatch for {type(pot).__name__}: "
                    f"expected {(n_local,)}, got {linear_mask.shape}"
                )
            submask = active_mask[offset:offset + n_local]
            if np.any(submask & ~linear_mask):
                return False
            offset += n_local

        if offset != n_params:
            raise ValueError(
                f"Linear scan mismatch: consumed {offset} params from a {n_params}-parameter container"
            )
        return True
    
    def optimizer_accepts_hessian(self) -> bool:
        """Check if the optimizer's step method accepts a 'hessian' argument."""
        return hasattr(self.optimizer, 'step') and 'hessian' in self.optimizer.step.__code__.co_varnames

    @abstractmethod
    def step(self, batch: Dict[str, Any], apply_update: bool = True):
        """
        Perform one optimization step from a pre-built trainer batch.

        Parameters
        ----------
        batch : dict
            Subclass-specific batch payload produced by the workflow.
        apply_update : bool, optional
            If True, apply the optimizer update. If False, return dry-run outputs.

        Returns
        -------
        dict
            Subclass-specific output payload.
        """
        pass

