"""
FILE: quantum_temple_multiverse/integration/quantum_bridge.py
PURPOSE: Map between cognition amplitudes and qudit register amplitudes.
MATHEMATICAL CORE:
  - Cognition vector c ∈ C^K (normalized)
  - Qudit register ψ ∈ C^{d^n}, we embed c into first K components (K ≤ d^n)
  - Optional basis rotations (Phase, DFT) to align with Temple operators
INTEGRATION POINTS:
  - cognition.consciousness_metrics (coherence/awareness)
  - qudit operators (DFT, Phase) for alignment
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class QuditCognitionBridge:
    d: int = 5
    n: int = 3  # D = d**n
    seed: int = 0

    def __post_init__(self):
        self.D = self.d ** self.n
        self.rng = np.random.default_rng(self.seed)

    def to_qudit(self, c: np.ndarray) -> np.ndarray:
        """Embed cognitive vector (K) into ψ (D) with K ≤ D."""
        c = np.asarray(c, dtype=complex)
        K = c.size
        if K > self.D:
            raise ValueError("Cognition dimension exceeds qudit register capacity.")
        psi = np.zeros(self.D, dtype=complex)
        psi[:K] = c
        nrm = np.linalg.norm(psi)
        return psi if nrm < 1e-12 else psi / nrm

    def to_cognition(self, psi: np.ndarray, K: int) -> np.ndarray:
        """Project ψ (D) back to first K cognitive slots."""
        psi = np.asarray(psi, dtype=complex)
        if psi.size != self.D:
            raise ValueError("ψ dimension mismatch.")
        c = psi[:K].copy()
        nrm = np.linalg.norm(c)
        return c if nrm < 1e-12 else c / nrm

    def align_phase(self, v: np.ndarray, theta: float = 0.0) -> np.ndarray:
        """Global phase rotate for operator alignment."""
        return v * np.exp(1j * theta)

if __name__ == "__main__":
    br = QuditCognitionBridge(d=5, n=3)
    c = (np.ones(20) + 1j) / np.sqrt(40)
    psi = br.to_qudit(c)
    c2  = br.to_cognition(psi, K=20)
    print("||psi||≈", round(np.linalg.norm(psi),4), "||diff||≈", round(np.linalg.norm(c2-c),6))
