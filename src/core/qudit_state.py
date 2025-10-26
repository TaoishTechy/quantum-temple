# src/core/qudit_state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

DEFAULT_CAP_NODES = 10_000  # hard safety cap (memory-friendly)

@dataclass
class QuditState:
    """Lightweight 'qudit' ensemble: per-node phase and amplitude."""
    n: int
    seed: int = 0
    amplitude: float = 1.0
    phases: np.ndarray = field(init=False, repr=False)
    amps:   np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.n <= 0 or self.n > DEFAULT_CAP_NODES:
            raise ValueError(f"n must be 1..{DEFAULT_CAP_NODES}")
        rng = np.random.default_rng(self.seed)
        self.phases = rng.uniform(-np.pi, np.pi, size=self.n).astype(np.float64)
        self.amps = np.full(self.n, float(self.amplitude), dtype=np.float64)

    def copy(self) -> "QuditState":
        out = QuditState(self.n, seed=self.seed, amplitude=self.amplitude)
        out.phases = self.phases.copy()
        out.amps = self.amps.copy()
        return out

    # Metrics
    def plv(self) -> float:
        """Phase Locking Value (0..1)."""
        return float(np.abs(np.mean(np.exp(1j * self.phases))))

    def variance(self) -> float:
        return float(np.var(self.phases))

    # Basic ops
    def add_phase(self, delta: float | np.ndarray) -> None:
        """Add phase in-place; supports scalar or per-node array."""
        self.phases = np.mod(self.phases + delta + np.pi, 2*np.pi) - np.pi

    def normalize(self) -> None:
        """Keep amplitudes bounded (L2)"""
        norm = np.linalg.norm(self.amps)
        if norm > 1e-12:
            self.amps /= norm
