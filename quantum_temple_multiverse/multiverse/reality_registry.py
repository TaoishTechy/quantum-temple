"""
FILE: quantum_temple_multiverse/multiverse/reality_registry.py
PURPOSE: Track and manage realities: add/get/list/remove with stability fields.
MATHEMATICAL CORE: Stores per-reality fitness F_reality, coherence, complexity, novelty.
INTEGRATION POINTS: multiverse.core, entities.*, civilizations.*, cognition.*, expansion.*, mathematics.multiversal_metrics
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class Reality:
    rid: str
    physical_constants: Dict[str, float]
    narrative_seed: Dict[str, Any]
    consciousness_template: Dict[str, Any]
    state: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=lambda: dict(
        fitness=0.0, complexity=0.0, stability=1.0, novelty=0.0, coherence=0.0
    ))

class RealityRegistry:
    def __init__(self):
        self._realities: Dict[str, Reality] = {}

    def add(self, r: Reality) -> None:
        if r.rid in self._realities:
            raise ValueError(f"Reality '{r.rid}' already exists.")
        self._realities[r.rid] = r

    def get(self, rid: str) -> Optional[Reality]:
        return self._realities.get(rid)

    def remove(self, rid: str) -> None:
        self._realities.pop(rid, None)

    def list_ids(self):
        return list(self._realities.keys())

    def snapshot(self):
        return {k: v.metrics for k, v in self._realities.items()}

if __name__ == "__main__":
    rr = RealityRegistry()
    rr.add(Reality("R1", {"c":1.0}, {"archetype":"seed"}, {"psi0":0.1}))
    print("Realities:", rr.list_ids())
    print("Snapshot:", rr.snapshot())
