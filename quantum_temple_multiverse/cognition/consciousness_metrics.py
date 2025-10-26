"""
FILE: quantum_temple_multiverse/cognition/consciousness_metrics.py
PURPOSE: Awareness & coherence measurements for cognitive states.
MATHEMATICAL CORE:
  Awareness ~ |∑ w_k e^{i arg(ψ_k)}|
  Coherence ~ PLV(ψ) = |mean e^{i arg(ψ)}|
INTEGRATION POINTS: multiverse.core, civilizations.symbolic_observatory
"""
from __future__ import annotations
import numpy as np

def awareness(psi: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones_like(psi, dtype=float)
    w = weights / (weights.sum() + 1e-12)
    phase = np.angle(psi + 1e-12)
    return float(np.abs(np.sum(w * np.exp(1j * phase))))

def coherence(psi: np.ndarray) -> float:
    phase = np.angle(psi + 1e-12)
    return float(np.abs(np.mean(np.exp(1j * phase))))

def purity_proxy(psi: np.ndarray) -> float:
    # 1 - variance of magnitudes (bounded 0..1)
    x = np.abs(psi)
    v = np.var(x)
    return float(max(0.0, 1.0 - v))

if __name__ == "__main__":
    a = np.ones(6, dtype=complex); a /= np.linalg.norm(a)
    print(dict(aw=awareness(a), coh=coherence(a), pur=purity_proxy(a)))
