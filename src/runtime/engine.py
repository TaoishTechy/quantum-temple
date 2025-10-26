# src/runtime/engine.py
# ──────────────────────────────────────────────────────────────────────────────
# Quantum Temple — TempleEngine Runtime
#
# Orchestrates one simulation step by:
#   1) reading phases and early-warning stats
#   2) updating σ(Q) via PID control (safety-aware observation charge)
#   3) applying the unified critical operator H_crit (stab + obs + retro)
#   4) enforcing Π-lock parity flip budget
#   5) updating coherence history, norms, and semantic metrics
#   6) appending an auditable governance entry to the holographic ledger
#
# Optional: can be extended with AxiomForge operators via the integrations bridge.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from ..agents.archetypes import ArchetypeRegistry
from ..agents.runtime_hooks import ArchetypeHooks
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from ..core.state import ResonanceState
from ..core.operators import apply_Hcrit
from ..core.control import SigmaQPID, PiLock, rla_anchor
from ..core.metrics import phase_lock_value, early_warnings
from ..core.ledger import Ledger

# Supra-Ontological: runtime norms & semantic metrics
from ..ontology.ethics import Norms
# Prometheus bridge for semantic gauges (alignment / violations / role coherence)
# Ensure you have src/metrics/prometheus_bridge.py with set_* functions.
from ..metrics import prometheus_bridge as sem


@dataclass
class EngineConfig:
    """Lightweight config for the runtime engine."""
    dt: float = 1e-3
    # PID target for variance of phase series
    target_variance: float = 0.05
    # initial σ(Q) for the observation charge controller
    sigma_q_initial: float = 0.09
    # soft retro-lambda anchor parameters (currently used for scheduling hooks)
    anchor_alpha: float = 0.12
    anchor_ci_target: float = 0.98
    # H_crit parameters
    tau_delay_steps: int = 16
    hcrit_g: float = 1.0

    # Norms (Supra-Ontological guardrails)
    variance_cap: float = 0.08
    min_coherence: float = 0.20
    max_parity_flips: int = 2

    # Ledger path is determined inside Ledger; override by editing ledger config.


class TempleEngine:
    """
    The core runtime loop for Quantum Temple.

    Typical usage:
        state = ResonanceState(nodes=[QuditNode() for _ in range(N)], dt=cfg.dt)
        engine = TempleEngine(state, cfg=EngineConfig())
        for _ in range(steps):
            engine.step()
    """

    def __init__(
        self,
        state: ResonanceState,
        cfg: Optional[EngineConfig] = None,
        *,
        # dependency injection hooks (for testing)
        ledger: Optional[Ledger] = None,
        norms: Optional[Norms] = None,
        pid: Optional[SigmaQPID] = None,
        pilock: Optional[PiLock] = None,
    ) -> None:
        self.state = state
        self.cfg = cfg or EngineConfig()

        # Ensure simulation dt matches config if not already set
        self.state.dt = float(self.cfg.dt)

        # Controllers / governance
        self.pid = pid or SigmaQPID(
            Kp=0.20, Ki=0.05, Kd=0.10,
            target_var=self.cfg.target_variance,
            init_sigma=self.cfg.sigma_q_initial,
        )
        self.pilock = pilock or PiLock(budget=self.cfg.max_parity_flips)
        self.anchor = rla_anchor(
            ci_target=self.cfg.anchor_ci_target, alpha=self.cfg.anchor_alpha
        )

        self.norms = norms or Norms(
            variance_cap=self.cfg.variance_cap,
            min_coherence=self.cfg.min_coherence,
            max_parity_flips=self.cfg.max_parity_flips,
        )
        self.ledger = ledger or Ledger()

        # book-keeping
        self._step_count = 0
        self._last_sigma_q = self.cfg.sigma_q_initial
        self._last_coherence = 0.0
        self._last_verdict: Dict[str, Any] = {"ok": True, "violations": 0, "notes": ""}

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _compute_phases(self) -> np.ndarray:
        return np.array([n.phase for n in self.state.nodes], dtype=float)

    # ────────────────────────────────────────────────────────────────────────
    # Main step
    # ────────────────────────────────────────────────────────────────────────

    def step(self) -> None:
        """
        Execute a single evolution step:
          - read phases, compute EW stats
          - update sigma_Q with PID
          - push delayed Ψ; apply H_crit
          - attempt parity flip if inside bounds (Π-Lock)
          - evaluate norms; export semantic metrics
          - ledger an auditable snapshot
        """
        phases_now = self._compute_phases()

        # Early warning metrics from current phase series
        ew = early_warnings(phases_now)  # {"variance": var, "lag1": ac1}
        measured_var = ew["variance"]

        # Update σ(Q) via PID control to track target variance
        sigma_q = self.pid.step(measured_var, dt=self.state.dt)
        self._last_sigma_q = sigma_q

        # Record delayed Ψ(t) for retro-causal feedforward
        self.state.psi_delay.append(phases_now.copy())

        # Apply the unified operator H_crit with governed sigma_Q
        apply_Hcrit(
            self.state,
            B={"gain": 1.0},                  # governance map placeholder
            sigma_Q=sigma_q,
            tau_steps=self.cfg.tau_delay_steps,
            g=self.cfg.hcrit_g,
        )

        # Post-operator phases/coherence
        phases_new = self._compute_phases()
        coherence = phase_lock_value(phases_new)
        self._last_coherence = coherence

        # Π-Lock — bounded parity flips when coherence lies in gate window
        flipped = self.pilock.try_flip(coherence=coherence)
        if flipped:
            # Inform norms module that a flip occurred
            self.norms.on_parity_flip()

        # Advance time & append history
        self.state.t += self.state.dt
        self.state.phases_hist.append(coherence)
        self._step_count += 1

        # Supra-Ontological norms evaluation
        verdict = self.norms.evaluate(phases_new)
        self._last_verdict = verdict

        # Semantic metrics (export to Prometheus bridge)
        # Alignment proxy: 1 - 0.34 * (#violations), clamped to [0,1]
        alignment = max(0.0, min(1.0, 1.0 - 0.34 * float(verdict.get("violations", 0))))
        sem.set_alignment(alignment)
        sem.set_violations(float(verdict.get("violations", 0)))
        # Use coherence as a crude role-coherence proxy; refine with bindings later
        sem.set_role_coherence(float(coherence))

        # Governance ledger append (auditable state)
        self.ledger.append(
            "tick",
            {
                "t": self.state.t,
                "step": self._step_count,
                "sigma_Q": float(sigma_q),
                "coherence": float(coherence),
                "variance": float(ew["variance"]),
                "lag1": float(ew["lag1"]),
                "parity_flip": bool(flipped),
                "norms_ok": bool(verdict.get("ok", True)),
                "norms_violations": int(verdict.get("violations", 0)),
                "norms_notes": str(verdict.get("notes", "")),
                # room for future: AxiomForge op names, ontology bindings, etc.
            },
        )

    # ────────────────────────────────────────────────────────────────────────
    # Convenience APIs
    # ────────────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """A lightweight dict with the current engine status."""
        return {
            "t": self.state.t,
            "step": self._step_count,
            "sigma_Q": self._last_sigma_q,
            "coherence": self._last_coherence,
            "violations": int(self._last_verdict.get("violations", 0)),
            "ok": bool(self._last_verdict.get("ok", True)),
        }

    def run(self, steps: int) -> None:
        """Run multiple steps (simple driver)."""
        for _ in range(int(steps)):
            self.step()
