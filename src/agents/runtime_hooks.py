from __future__ import annotations
from typing import List
import numpy as np
from ..core.state import ResonanceState
from .archetypes import ArchetypeRegistry, ArchetypeSpec

class ArchetypeHooks:
    """
    Applies archetype-specific adjustments at runtime:
      - tune PID targets (sigma_Q target variance) per archetype
      - gate parity flips by budget
      - (optional) route to symbolic operators by name
    """
    def __init__(self, registry: ArchetypeRegistry):
        self.registry = registry
        # counters per archetype
        self._flip_counts = {}

    def on_step_pre(self, state: ResonanceState, engine) -> None:
        # adjust controller targets per node archetype (average across nodes)
        # Example: update PID target variance as the mean of bound archetypes
        targets = []
        for i, _ in enumerate(state.nodes):
            at = self.registry.get_archetype_for_node(i)
            if at and "sigma_q_target_variance" in at.control:
                targets.append(float(at.control["sigma_q_target_variance"]))
        if targets:
            engine.pid.target = float(np.mean(targets))

    def on_parity_flip(self, node_indices: List[int], engine) -> bool:
        # enforce archetype-specific parity flip budgets
        # if any node belongs to an archetype with zero budget, deny
        for idx in node_indices:
            at = self.registry.get_archetype_for_node(idx)
            if not at: continue
            budget = int(at.control.get("parity_flip_budget", engine.pilock.budget))
            used = self._flip_counts.get(at.name, 0)
            if used >= budget:
                return False
        # allow flip and update counters
        for idx in node_indices:
            at = self.registry.get_archetype_for_node(idx)
            if at:
                self._flip_counts[at.name] = self._flip_counts.get(at.name, 0) + 1
        return True
