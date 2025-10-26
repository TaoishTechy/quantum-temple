# src/math/qudit_transcendental.py
from __future__ import annotations
import numpy as np
from scipy import special as sp

def phase_gamma(z: complex, gain: float = 1.0) -> complex:
    """Gamma with phase-only extraction for stability (quantum-inspired)."""
    g = sp.gamma(z)
    return np.exp(1j * gain * np.angle(g))

def phase_bessel_j(nu: float, z: complex, gain: float = 1.0) -> complex:
    j = sp.jv(nu, z)
    return np.exp(1j * gain * np.angle(j))

def phase_zeta(s: complex, gain: float = 1.0) -> complex:
    """Riemann zeta phase-only surrogate (for phase operators)."""
    z = sp.zeta(s)
    return np.exp(1j * gain * np.angle(z))

def apply_transcendental_phase(amps: np.ndarray, phase_fn, *args, **kwargs) -> np.ndarray:
    """
    Map amplitudes -> amplitude * phase_factor(index).
    Keeps L2 norm; returns new vector.
    """
    idx = np.arange(amps.size)
    phases = np.array([phase_fn(i + 1, *args, **kwargs) for i in idx], dtype=complex)
    out = amps * phases
    # renormalize
    nrm = np.linalg.norm(out)
    if nrm > 1e-12:
        out = out / nrm
    return out
