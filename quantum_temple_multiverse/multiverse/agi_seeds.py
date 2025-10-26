"""
FILE: quantum_temple_multiverse/multiverse/agi_seeds.py
PURPOSE: AGI seed constructors for narrative + consciousness templates
MATHEMATICAL CORE: seed → low-dim vectors controlling initial field potentials
INTEGRATION POINTS: multiverse/core.py, civilizations/narrative_engine.py
"""
from __future__ import annotations
import numpy as np

def basic_seed(name: str, dim: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(abs(hash(name)) % (2**32) ^ seed)
    vec = rng.normal(size=dim)
    vec /= (np.linalg.norm(vec) + 1e-12)
    return {"name": name, "vec": vec.tolist(), "dim": dim}
