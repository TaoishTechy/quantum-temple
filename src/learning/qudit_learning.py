# src/learning/qudit_learning.py
from __future__ import annotations
import numpy as np
from ..cognition.qudit_cognition import CognitiveState

class HebbianLearning:
    """
    Quantum-inspired Hebbian rule: ΔW = η · ψ ψ*  (outer product of belief amps)
    W stores complex associations between concepts; bounded norm.
    """
    def __init__(self, K: int, lr: float = 0.02, max_norm: float = 3.0):
        self.W = np.zeros((K, K), dtype=complex)
        self.lr = float(lr)
        self.max_norm = float(max_norm)

    def step(self, cs: CognitiveState) -> None:
        psi = cs.belief
        self.W += self.lr * np.outer(psi, np.conjugate(psi))
        n = np.linalg.norm(self.W)
        if n > self.max_norm:
            self.W *= (self.max_norm / n)

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Associative recall: y = W * cue (complex)."""
        return self.W @ cue

def consolidate_memory(cs: CognitiveState, strength: float = 0.05) -> None:
    """
    MemoryConsolidation: purification toward dominant component; gentle.
    """
    p = np.abs(cs.belief) ** 2
    j = int(np.argmax(p))
    proj = np.zeros_like(cs.belief); proj[j] = 1.0 + 0j
    cs.belief = (1 - strength) * cs.belief + strength * proj
    cs.belief /= (np.linalg.norm(cs.belief) + 1e-12)

def forgetting_curve(cs: CognitiveState, lam: float = 0.01) -> None:
    """Ebbinghaus-style amplitude decay."""
    cs.belief *= (1.0 - lam)
    n = np.linalg.norm(cs.belief)
    if n < 1e-12:
        # re-inject tiny white amplitude to avoid dead state
        cs.belief += 1e-6
    cs.belief /= (np.linalg.norm(cs.belief) + 1e-12)
