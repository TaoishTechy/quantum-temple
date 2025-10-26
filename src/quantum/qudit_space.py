# src/quantum/qudit_space.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple
import numpy as np

Array = np.ndarray

def _normalize(psi: Array) -> Array:
    nrm = np.linalg.norm(psi)
    if nrm > 1e-12:
        psi = psi / nrm
    return psi

@dataclass
class Qudit:
    """Single qudit (local) — convenience for small demos."""
    d: int = 5
    state: Array = None

    def __post_init__(self):
        if self.state is None:
            self.state = np.zeros(self.d, dtype=complex)
            self.state[0] = 1.0 + 0j
        else:
            self.state = _normalize(np.asarray(self.state, dtype=complex))
        if self.state.shape != (self.d,):
            raise ValueError("state must be shape (d,)")

    def apply_gate(self, U: Array) -> None:
        U = np.asarray(U, dtype=complex)
        if U.shape != (self.d, self.d):
            raise ValueError("gate must be (d,d)")
        self.state = _normalize(U @ self.state)


class QuantumRegister:
    """
    Dense register (explicit state vector of size D=d**n). Use for small n.
    For Temple's big-N phase engine use phase-space classes elsewhere.
    """
    def __init__(self, n_qudits: int, d: int = 5, seed: Optional[int] = None):
        if n_qudits <= 0:
            raise ValueError("n_qudits must be >= 1")
        self.n = n_qudits
        self.d = d
        self.D = d ** n_qudits
        self.rng = np.random.default_rng(seed)
        self.state: Array = np.zeros(self.D, dtype=complex)
        self.state[0] = 1.0 + 0j

    # ---- helpers ---------------------------------------------------------
    def _reshape_tensor(self) -> Array:
        return self.state.reshape((self.d,) * self.n)

    def _apply_single_qudit(self, U: Array, target: int) -> None:
        """
        Apply (d,d) gate U to target qudit (0..n-1).
        Tensor trick: move axis, matmul, move back (no full kron build).
        """
        if U.shape != (self.d, self.d):
            raise ValueError("U must be (d,d)")
        psi = self._reshape_tensor()
        # bring target axis to front
        axes = list(range(self.n))
        axes[0], axes[target] = axes[target], axes[0]
        psi = np.transpose(psi, axes)
        # matmul over leading axis
        psi = np.tensordot(U, psi, axes=([1], [0]))
        # undo axes
        psi = np.transpose(psi, np.argsort(axes))
        self.state = _normalize(psi.reshape(self.D))

    def _apply_two_qudit(self, U: Array, a: int, b: int) -> None:
        """
        Apply (d*d, d*d) gate to two targets (a,b). Uses reshape+einsum.
        """
        if U.shape != (self.d * self.d, self.d * self.d):
            raise ValueError("U must be (d*d, d*d)")
        if a == b:
            raise ValueError("distinct targets required")

        psi = self._reshape_tensor()
        # bring (a,b) to the front as two axes
        axes = list(range(self.n))
        axes[0], axes[a] = axes[a], axes[0]
        axes[1], axes[b] = axes[b], axes[1]
        psi = np.transpose(psi, axes)

        psi2 = psi.reshape(self.d * self.d, -1)
        psi2 = U @ psi2
        psi = psi2.reshape((self.d, self.d) + psi.shape[2:])
        # undo axes
        inv = np.argsort(axes)
        psi = np.transpose(psi, inv)
        self.state = _normalize(psi.reshape(self.D))

    # ---- public API ------------------------------------------------------
    def apply_gate(self, U: Array, target: int) -> None:
        self._apply_single_qudit(U, target)

    def apply_two_qudit(self, U: Array, a: int, b: int) -> None:
        self._apply_two_qudit(U, a, b)

    def measure_probabilities(self) -> Array:
        return np.abs(self.state) ** 2

    def sample(self, shots: int = 1) -> np.ndarray:
        p = self.measure_probabilities()
        return self.rng.choice(self.D, size=shots, p=p)
