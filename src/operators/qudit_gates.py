# src/operators/qudit_gates.py
from __future__ import annotations
import numpy as np

def X(d: int) -> np.ndarray:
    """Generalized shift (cyclic) gate."""
    U = np.zeros((d, d), dtype=complex)
    for i in range(d):
        U[(i + 1) % d, i] = 1.0
    return U

def Z(d: int) -> np.ndarray:
    """Generalized phase gate."""
    omega = np.exp(2j * np.pi / d)
    return np.diag([omega ** k for k in range(d)])

def DFT(d: int) -> np.ndarray:
    """Generalized Hadamard (discrete Fourier) gate."""
    idx = np.arange(d)
    U = np.exp(2j * np.pi * np.outer(idx, idx) / d) / np.sqrt(d)
    return U

def Phase(d: int, theta: float) -> np.ndarray:
    """Single-qudit uniform phase rotation (e^{i theta k})."""
    k = np.arange(d)
    return np.diag(np.exp(1j * theta * k))

def Rz(d: int, theta: float) -> np.ndarray:
    """Z-axis-like rotation: exp(i theta * diag(k - (d-1)/2))."""
    k = np.arange(d) - (d - 1) / 2.0
    return np.diag(np.exp(1j * theta * k))

def Rx(d: int, theta: float) -> np.ndarray:
    """
    X-axis-like rotation via exp(i theta * (X + X^†)/2).
    Uses truncated series; small theta recommended.
    """
    Xd = X(d)
    H = 0.5 * (Xd + Xd.conj().T)
    # 2nd-order Taylor (sufficient for small theta; CPU-only)
    I = np.eye(d, dtype=complex)
    return I + 1j * theta * H - 0.5 * (theta ** 2) * (H @ H)

def CADD(d: int) -> np.ndarray:
    """
    Controlled-add: |a,b> -> |a, (a+b) mod d>
    Two-qudit entangler (d*d, d*d).
    """
    D = d * d
    U = np.zeros((D, D), dtype=complex)
    for a in range(d):
        for b in range(d):
            i = a * d + b
            j = a * d + ((a + b) % d)
            U[j, i] = 1.0
    return U
