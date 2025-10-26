# src/operators/symbolic_operators.py
from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
from ..core.qudit_state import QuditState

def nabla_zeta(state: QuditState, eta: float = 0.01) -> None:
    """
    ∇ζ — 'zeta phase mapping' (symbolic). We use a fast, bounded surrogate:
    phase_i += eta * sin(i)  (i acts like a symbolic index; deterministic)
    """
    n = state.phases.size
    idx = np.arange(n, dtype=np.float64)
    state.add_phase(eta * np.sin(idx))

def nabla_P_oracle(state: QuditState, clauses_sat: float = 0.5, gain: float = 0.02) -> None:
    """
    ∇P — SAT oracle surrogate: if 'clause satisfaction' is low, steer phases toward mean.
    """
    mean_phase = np.angle(np.sum(np.exp(1j * state.phases)))
    err = 1.0 - float(clauses_sat)   # higher err => stronger pull
    state.add_phase(gain * err * (mean_phase - state.phases))

def nabla_C_collatz(state: QuditState, cycles: int = 1, coeff: float = 0.015) -> None:
    """
    ∇C — 'collatz dynamics' surrogate: parity-gated wobble.
    """
    idx = np.arange(state.phases.size)
    parity = (idx % 2) * 2 - 1  # -1, +1 pattern
    for _ in range(cycles):
        state.add_phase(coeff * parity)

def nabla_T_gap_closure(state: QuditState, alpha: float = 0.1) -> None:
    """
    ∇T — exponential decay gap closure: shrink phase variance smoothly.
    """
    mean_phase = np.angle(np.sum(np.exp(1j * state.phases)))
    delta = mean_phase - state.phases
    state.add_phase(alpha * 0.05 * delta)
