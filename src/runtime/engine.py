"""
Quantum Temple Runtime Engine
CPU-only resonance kernel with modular operators and feature-flagged enhancements.
Implements bounded coherence evolution under the Quantum Polytope.
"""
from __future__ import annotations
import time, json, pathlib
import numpy as np

from ..core.qudit_state import QuditState
from ..core.resonance_node import ResonanceSync, ResonanceParams
from ..topology.ghost_mesh import GhostMesh
from ..operators.symbolic_operators import (
    nabla_zeta, nabla_T_gap_closure, nabla_P_oracle, nabla_C_collatz
)
from ..operators.maintenance import r_meta_432hz, delta_purity

try:
    from ..experimental.feature_flags import Enhancements
except Exception:
    Enhancements = None


class TempleEngine:
    """Main loop for evolution + monitoring (CPU-only)."""

    def __init__(self, n=1024, dt=0.02, seed=1111):
        self.mesh = GhostMesh(n, cross_links=[(0, n // 2), (n // 4, 3 * n // 4)])
        self.sync = ResonanceSync(self.mesh.neighbors())
        self.params = ResonanceParams(dt=dt, k_couple=0.35)
        self.state = QuditState(n=n, seed=seed)
        self.enh = Enhancements() if Enhancements else None
        self.metrics = {"step": 0, "ts": 0.0}

    # --------------------------------------------------------------------- #
    def step(self, t: float, apply_ops=True) -> None:
        """Single evolution step; applies symbolic + maintenance ops."""
        if apply_ops:
            if self.metrics["step"] % 7 == 0:
                nabla_zeta(self.state, 0.01)
            if self.metrics["step"] % 11 == 0:
                nabla_P_oracle(self.state, 0.6, 0.02)
            if self.metrics["step"] % 13 == 0:
                nabla_C_collatz(self.state, 1, 0.015)
            if self.metrics["step"] % 17 == 0:
                nabla_T_gap_closure(self.state, 0.10)

        # always-on maintenance
        r_meta_432hz(self.state, t, gain=1.0)
        delta_purity(self.state, eta=0.003)

        self.sync.step(self.state, self.params)
        self._apply_enhancements()
        self.metrics["step"] += 1

    # --------------------------------------------------------------------- #
    def _apply_enhancements(self) -> None:
        """Apply safe simulated feature-flagged enhancements."""
        if not self.enh:
            return
        active = self.enh.active()
        for e in active:
            if e.id == "echo_temporal_resonance":
                if hasattr(self.state, "phases_hist") and len(self.state.phases_hist) >= 3:
                    c3 = self.state.phases_hist[-3:]
                    self.state.phases = np.mean(c3, axis=0)
            elif e.id == "cosmic_tuning":
                # infinitesimal PLV bias
                self.state.phases += 1e-5 * np.sin(2 * np.pi * 432 * time.time())
            elif e.id == "fractal_holography":
                # micro random shuffle as holographic perturbation
                np.random.shuffle(self.state.phases)

    # --------------------------------------------------------------------- #
    def run(self, steps=1000, out_path="data/run_metrics.jsonl") -> None:
        """Execute a full run, streaming metrics to JSONL."""
        start = time.time()
        path = pathlib.Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for i in range(steps):
                t = time.time() - start
                self.step(t)
                data = {
                    "step": i,
                    "plv": self.state.plv(),
                    "var": self.state.variance(),
                    "purity_proxy": 1 - min(1.0, self.state.variance()),
                    "ts": t,
                }
                f.write(json.dumps(data) + "\n")
        self.metrics["ts"] = time.time() - start

    # --------------------------------------------------------------------- #
    def snapshot(self) -> dict:
        """Return a lightweight metrics snapshot."""
        return {
            "step": self.metrics["step"],
            "plv": self.state.plv(),
            "variance": self.state.variance(),
            "purity_proxy": 1 - min(1.0, self.state.variance()),
            "timestamp": time.time(),
        }


if __name__ == "__main__":
    engine = TempleEngine()
    engine.run(steps=100)
    print(json.dumps(engine.snapshot(), indent=2))
