"""
FILE: quantum_temple_multiverse/multiverse/reality_registry.py
PURPOSE: Manage realities, store states, serialize/deserialize
MATHEMATICAL CORE: stores per-reality narrative params, physical state, conscious field structure
INTEGRATION POINTS: multiverse/core.py, entities/drift_resonance.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterator, Any, List
import itertools, numpy as np

from ..mathematics.quantum_consciousness import ConsciousField

_uid_counter = itertools.count(1)

@dataclass
class Reality:
    uid: int
    narrative_params: np.ndarray
    physical_state: np.ndarray
    conscious_field: ConsciousField
    civilization: Dict[str, Any]
    fitness: float = 0.0
    log: List[Dict[str, Any]] = field(default_factory=list)

class RealityRegistry:
    def __init__(self):
        self._realities: Dict[int, Reality] = {}

    def create_reality(self, narrative_params, physical_state, conscious_field, civ) -> Reality:
        uid = next(_uid_counter)
        R = Reality(uid, np.array(narrative_params, float), np.array(physical_state, float), conscious_field, civ)
        self._realities[uid] = R
        return R

    def get(self, uid: int) -> Reality:
        return self._realities[uid]

    def __iter__(self) -> Iterator[Reality]:
        return iter(self._realities.values())

    def as_matrix(self) -> np.ndarray:
        """Simple matrix of pairwise overlaps of narrative params."""
        keys = list(self._realities.keys())
        if not keys:
            return np.zeros((0,0))
        P = np.stack([self._realities[k].narrative_params for k in keys], axis=0)
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        return Pn @ Pn.T

    def physical_state_from_constants(self, consts: Dict[str, float]) -> np.ndarray:
        v = np.array([float(consts.get(k,0.0)) for k in sorted(consts.keys())], float)
        if v.size == 0: v = np.array([1.0], float)
        return v

    def log_event(self, uid: int, event: Dict[str, Any]) -> None:
        self._realities[uid].log.append(event)

    def serialize(self) -> Dict[str, Any]:
        out = {}
        for uid, R in self._realities.items():
            out[uid] = {
                "narrative_params": R.narrative_params.tolist(),
                "physical_state": R.physical_state.tolist(),
                "conscious_field": R.conscious_field.to_dict(),
                "civilization": R.civilization,
                "fitness": R.fitness,
                "log": R.log[-16:],
            }
        return out
