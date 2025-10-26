# src/cognition/qudit_cognition.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

Array = np.ndarray

def _normalize(v: Array) -> Array:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

@dataclass
class AttentionVector:
    """
    Attention weights over concepts/basis, sum=1.0, nonnegative.
    Used for focus/spotlight and Global Workspace selection.
    """
    weights: Array  # shape (K,)
    def __post_init__(self):
        w = np.asarray(self.weights, dtype=float)
        w = np.clip(w, 0.0, np.inf)
        s = w.sum()
        self.weights = w if s < 1e-12 else w / s

    def sharpen(self, kappa: float = 1.2) -> "AttentionVector":
        w = self.weights ** float(kappa)
        s = w.sum()
        return AttentionVector(w if s < 1e-12 else w / s)

@dataclass
class WorkingMemory:
    """
    Small, bounded buffer of concept amplitudes with exponential decay.
    """
    size: int
    decay: float = 0.02
    state: Array = field(init=False)

    def __post_init__(self):
        self.state = np.zeros(self.size, dtype=complex)

    def write(self, idx: int, amp: complex) -> None:
        self.state[idx % self.size] = amp

    def step(self) -> None:
        self.state *= (1.0 - self.decay)

@dataclass
class CognitiveState:
    """
    Belief amplitude over K concepts (not a full register; lean).
    """
    d: int = 5
    K: int = 25
    seed: int = 0
    belief: Array = field(init=False)     # shape (K,) complex amps
    basis_map: Dict[int, int] = field(default_factory=dict)  # concept -> qudit basis index [0..d-1]

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        z = rng.normal(size=self.K) + 1j * rng.normal(size=self.K)
        self.belief = _normalize(z)

    @property
    def probs(self) -> Array:
        p = np.abs(self.belief) ** 2
        s = p.sum()
        return p if s < 1e-12 else p / s

    def coherence(self) -> float:
        # simple PLV-like scalar for cognition
        ang = np.angle(self.belief + 1e-12)
        return float(np.abs(np.mean(np.exp(1j * ang))))

    def cognitive_dissonance(self, attention: AttentionVector) -> float:
        """
        Dissonance as mismatch between belief distribution and attention focus.
        """
        p = self.probs
        a = attention.weights
        # Jensen-Shannon style bounded divergence proxy
        m = 0.5 * (p + a)
        eps = 1e-12
        kl_pm = (p * (np.log(p + eps) - np.log(m + eps))).sum()
        kl_am = (a * (np.log(a + eps) - np.log(m + eps))).sum()
        js = 0.5 * (kl_pm + kl_am)
        return float(js)

    def spotlight(self, attention: AttentionVector, gain: float = 0.06) -> None:
        """
        SpotlightAttention: amplify attended concepts via phase-preserving scaling.
        """
        scale = 1.0 + gain * (attention.weights - attention.weights.mean())
        self.belief = _normalize(self.belief * scale)

    def collapse_to_focus(self, attention: AttentionVector, rng=None) -> int:
        """
        Measurement-like readout in the attention-chosen basis.
        """
        rng = rng or np.random.default_rng(self.seed)
        idx = int(rng.choice(self.K, p=self.probs))
        # project toward focused index while preserving superposition softly
        alpha = 0.25 + 0.5 * attention.weights[idx]
        proj = np.zeros_like(self.belief)
        proj[idx] = 1.0 + 0j
        self.belief = _normalize(alpha * proj + (1.0 - alpha) * self.belief)
        return idx
