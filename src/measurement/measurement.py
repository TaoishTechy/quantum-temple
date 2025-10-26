# src/measurement/measurement.py
from __future__ import annotations
import numpy as np
from typing import Tuple

def probabilities(state: np.ndarray) -> np.ndarray:
    p = np.abs(state) ** 2
    s = p.sum()
    if s > 1e-12:
        p = p / s
    return p

def measure(state: np.ndarray, rng=None) -> Tuple[int, np.ndarray]:
    rng = rng or np.random.default_rng()
    p = probabilities(state)
    i = rng.choice(state.size, p=p)
    proj = np.zeros_like(state)
    proj[i] = 1.0 + 0j
    return i, proj

def expectation(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.vdot(state, operator @ state)))
