# src/metrics/consciousness_metrics.py
from __future__ import annotations
import numpy as np
from ..cognition.qudit_cognition import CognitiveState, AttentionVector

def awareness_metric(cs: CognitiveState, av: AttentionVector) -> float:
    """
    Awareness as coherence across attention-weighted concepts.
    """
    w = av.weights
    phase = np.angle(cs.belief + 1e-12)
    return float(np.abs(np.sum(w * np.exp(1j * phase))))

def cognitive_integration_I(cs: CognitiveState) -> float:
    """
    Simple IIT-flavored proxy: I = 1 - σ² over concept magnitudes.
    """
    x = np.abs(cs.belief)
    v = np.var(x)
    return float(max(0.0, 1.0 - v))

def dissonance_score(cs: CognitiveState, av: AttentionVector) -> float:
    """Expose CognitiveState.cognitive_dissonance for metrics pipelines."""
    return cs.cognitive_dissonance(av)

def cognitive_load_entropy(cs: CognitiveState, wm_state: np.ndarray) -> float:
    """
    Cognitive load ~ entanglement-like coupling between belief and WM.
    Here: mutual-info proxy using correlation of magnitudes.
    """
    a = np.abs(cs.belief); a = a / (a.sum() + 1e-12)
    b = np.abs(wm_state);  b = b / (b.sum() + 1e-12)
    m = 0.5 * (a + b)
    eps = 1e-12
    kl_am = np.sum(a * (np.log(a + eps) - np.log(m + eps)))
    kl_bm = np.sum(b * (np.log(b + eps) - np.log(m + eps)))
    js = 0.5 * (kl_am + kl_bm)
    return float(js)
