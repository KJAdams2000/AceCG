import numpy as np

from .base import BasePotential


class SoftPotential(BasePotential):
    """Cosine-soft pair potential ``U(r) = A * (1 + cos(pi * r / r_c))`` for ``r < r_c``.

    Parameters are ``[A, r_c]``. Only ``A`` is linear; ``r_c`` is nonlinear.
    """

    def __init__(self, typ1, typ2, A, cutoff):
        super().__init__()
        self.typ1 = typ1
        self.typ2 = typ2
        self._params = np.array([A, cutoff], dtype=float)
        self._params_to_scale = [0]
        self._param_names = ["A", "r_c"]
        self._dparam_names = ["dA", "drc"]
        self._d2param_names = [
            ["dA_2", "dAdrc"],
            ["dAdrc", "drc_2"],
        ]

    @property
    def cutoff(self) -> float:
        return float(self._params[1])

    def is_param_linear(self) -> np.ndarray:
        return np.array([True, False], dtype=bool)

    def value(self, r):
        r = np.asarray(r, dtype=float)
        A = float(self._params[0])
        cutoff = self.cutoff
        values = A * (1.0 + np.cos(np.pi * r / cutoff))
        return np.where(r < cutoff, values, 0.0)

    def force(self, r):
        r = np.asarray(r, dtype=float)
        A = float(self._params[0])
        cutoff = self.cutoff
        values = A * (np.pi / cutoff) * np.sin(np.pi * r / cutoff)
        return np.where(r < cutoff, values, 0.0)

    def dA(self, r):
        r = np.asarray(r, dtype=float)
        cutoff = self.cutoff
        values = 1.0 + np.cos(np.pi * r / cutoff)
        return np.where(r < cutoff, values, 0.0)

    def drc(self, r):
        r = np.asarray(r, dtype=float)
        A = float(self._params[0])
        cutoff = self.cutoff
        values = A * np.pi * r * np.sin(np.pi * r / cutoff) / (cutoff ** 2)
        return np.where(r < cutoff, values, 0.0)

    def dA_2(self, r):
        r = np.asarray(r, dtype=float)
        return np.zeros_like(r, dtype=float)

    def dAdrc(self, r):
        r = np.asarray(r, dtype=float)
        cutoff = self.cutoff
        values = np.pi * r * np.sin(np.pi * r / cutoff) / (cutoff ** 2)
        return np.where(r < cutoff, values, 0.0)

    def dA_drc(self, r):
        return self.dAdrc(r)

    def drc_2(self, r):
        r = np.asarray(r, dtype=float)
        A = float(self._params[0])
        cutoff = self.cutoff
        phase = np.pi * r / cutoff
        values = -A * (
            2.0 * np.pi * r * np.sin(phase) / (cutoff ** 3)
            + (np.pi ** 2) * (r ** 2) * np.cos(phase) / (cutoff ** 4)
        )
        return np.where(r < cutoff, values, 0.0)