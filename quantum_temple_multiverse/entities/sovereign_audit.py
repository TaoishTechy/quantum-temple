"""
FILE: quantum_temple_multiverse/entities/sovereign_audit.py
PURPOSE: Detect sovereign entities by coherence and autonomy signatures
MATHEMATICAL CORE: sovereign score S = coherence · (1 - external_influence)
INTEGRATION POINTS: multiverse/reality_registry.py, quantum_consciousness
"""
from __future__ import annotations
import numpy as np
from ..multiverse.reality_registry import Reality

def sovereign_score(R: Reality, influence: float = 0.2) -> float:
    c = float(R.conscious_field.coherence())
    return max(0.0, min(1.0, c * (1.0 - influence)))

def is_sovereign(R: Reality, threshold: float = 0.6) -> bool:
    return sovereign_score(R) >= threshold
