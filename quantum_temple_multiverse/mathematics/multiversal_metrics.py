"""
FILE: quantum_temple_multiverse/mathematics/multiversal_metrics.py
PURPOSE: Cross-reality fitness: F = ∫ complexity·stability·novelty dμ  (discrete sum)
MATHEMATICAL CORE: Weighted product with normalization & clipping.
INTEGRATION POINTS: multiverse.core, multiverse.reality_registry
"""
from __future__ import annotations
import numpy as np

def reality_fitness(complexity: float, stability: float, novelty: float, w=(1.0,1.0,1.0)) -> float:
    c = max(0.0, float(complexity))
    s = max(0.0, float(stability))
    n = max(0.0, float(novelty))
    wc, ws, wn = [max(0.0, float(x)) for x in w]
    score = (c**wc) * (s**ws) * (n**wn)
    return float(min(1e6, score))  # cap to avoid overflow

if __name__ == "__main__":
    print("F≈", reality_fitness(2.0, 0.9, 1.5, (1,2,1)))
