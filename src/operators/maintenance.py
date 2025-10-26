# CPU-only, in-place, O(N)
from __future__ import annotations
import numpy as np
from ..core.qudit_state import QuditState

def r_meta_432hz(state: QuditState, t: float, gain: float = 1.0) -> None:
    """
    R_meta(432Hz): deterministic low-frequency carrier (no audio IO).
    Adds a tiny, smooth phase modulation; bounded & reproducible.
    phase += gain * sin(2π * 432 * t) / (1 + e^{-E_coh})
    We approximate E_coh with PLV (0..1).
    """
    plv = state.plv()
    denom = 1.0 + np.exp(-max(0.0, min(1.0, plv)))
    bump = gain * np.sin(2.0 * np.pi * 432.0 * t) / denom
    state.add_phase(bump)

def delta_purity(state: QuditState, eta: float = 0.003) -> None:
    """
    Δ_purity: very gentle variance shrink toward circular mean (maintenance).
    Implements a tiny corrective bias; preserves wrap and stays O(N).
    """
    mean_phase = np.angle(np.sum(np.exp(1j * state.phases)))
    delta = mean_phase - state.phases
    state.add_phase(eta * 0.05 * delta)
