# src/operators/cognitive_operators.py
from __future__ import annotations
import numpy as np
from typing import Optional
from ..cognition.qudit_cognition import CognitiveState, AttentionVector

def Pi_attention(cs: CognitiveState, av: AttentionVector, focus_gain: float = 1.0) -> None:
    """
    Π_attention: measurement-basis selection pressure (focus amplification).
    """
    cs.spotlight(av, gain=0.04 * float(focus_gain))

def Phi_intention(cs: CognitiveState, target_idx: Optional[int], eta: float = 0.035) -> None:
    """
    Φ_intention: push phase of the intended concept forward; mild global bias.
    """
    if target_idx is None:
        return
    phase_bias = np.zeros_like(cs.belief)
    phase_bias[target_idx] = 1.0 + 0j
    # apply tiny phase-only rotation toward target
    ang = np.angle(cs.belief + 1e-12)
    ang = ang + eta * np.real(phase_bias)
    cs.belief = (np.abs(cs.belief) + 1e-12) * np.exp(1j * ang)
    cs.belief = cs.belief / (np.linalg.norm(cs.belief) + 1e-12)

def Omega_decision(cs: CognitiveState, utility: np.ndarray, beta: float = 4.0) -> int:
    """
    Ω_decision: softmax over utilities, with quantum interference proxy via phases.
    Returns chosen index; also nudges belief toward chosen concept.
    """
    u = np.asarray(utility, dtype=float)
    u = u - u.max()
    p = np.exp(beta * u); p = p / (p.sum() + 1e-12)
    # interference proxy: align phases with utility sign
    phase = np.sign(u + 1e-12) * np.pi * 0.05
    cs.belief *= np.exp(1j * phase)
    # collapse softly toward sample
    idx = int(np.random.choice(len(u), p=p))
    alpha = 0.35
    proj = np.zeros_like(cs.belief); proj[idx] = 1.0 + 0j
    cs.belief = (alpha * proj + (1.0 - alpha) * cs.belief)
    cs.belief = cs.belief / (np.linalg.norm(cs.belief) + 1e-12)
    return idx

def Phi_reflection(cs: CognitiveState, kappa: float = 0.15) -> None:
    """
    Φ_reflection: meta-cognitive phase smoothing (self-observation).
    """
    ang = np.angle(cs.belief + 1e-12)
    mean = np.angle(np.sum(np.exp(1j * ang)))
    ang = (1 - kappa) * ang + kappa * mean
    cs.belief = (np.abs(cs.belief) + 1e-12) * np.exp(1j * ang)
    cs.belief /= (np.linalg.norm(cs.belief) + 1e-12)
