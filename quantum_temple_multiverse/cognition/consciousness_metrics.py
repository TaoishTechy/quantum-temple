"""
FILE: quantum_temple_multiverse/cognition/consciousness_metrics.py
PURPOSE: Awareness/Coherence/Dissonance measurements
MATHEMATICAL CORE:
  Awareness A = |Σ w_k e^{iθ_k}|; Dissonance via Jensen–Shannon proxy
INTEGRATION POINTS: used by core and reporting dashboards
"""
from __future__ import annotations
import numpy as np

def awareness(psi: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones_like(psi, dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    theta = np.angle(psi + 1e-12)
    return float(np.abs(np.sum(weights * np.exp(1j * theta))))

def dissonance(p: np.ndarray, a: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    a = a / (a.sum() + 1e-12)
    m = 0.5 * (p + a)
    eps = 1e-12
    js = 0.5 * (np.sum(p * (np.log(p + eps) - np.log(m + eps))) +
                np.sum(a * (np.log(a + eps) - np.log(m + eps))))
    return float(js)
