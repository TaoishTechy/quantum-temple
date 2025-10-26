# src/algorithms/qudit_qft.py
from __future__ import annotations
import numpy as np

def qft_register(state: np.ndarray, d: int, n: int) -> np.ndarray:
    """
    QFT over radix-d register of n qudits (size D=d**n).
    Implemented with an FFT-like transform on length D, scaled by 1/sqrt(D).
    CPU-only; suitable for small n (demo/analysis).
    """
    D = d ** n
    if state.shape != (D,):
        raise ValueError("state must be (d**n,)")
    # Use dense DFT via FFT; ensure consistent scaling
    psi = np.fft.fft(state, n=D)
    psi = psi / np.sqrt(D)
    return psi

def iqft_register(state: np.ndarray, d: int, n: int) -> np.ndarray:
    D = d ** n
    if state.shape != (D,):
        raise ValueError("state must be (d**n,)")
    psi = np.fft.ifft(state, n=D) * np.sqrt(D)
    return psi
