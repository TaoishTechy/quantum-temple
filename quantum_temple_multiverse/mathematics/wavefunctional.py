"""
FILE: quantum_temple_multiverse/mathematics/wavefunctional.py
PURPOSE: Practical surrogate for |Ψ_multiverse⟩ sampling & scoring.
MATHEMATICAL CORE:
  Formal: |Ψ_M⟩ = ∫ D[g] D[Φ] D[Ψ_c] e^{i S_total[g,Φ,Ψ_c]} |g, Φ, Ψ_c⟩
  Here: we discretize (g, Φ, Ψ_c) to finite params θ and define:
    S_total(θ) = S_geom(θ_g) + S_field(θ_Φ) + S_conscious(Ψ_c)
  We sample θ via tempered Metropolis-Hastings and return (θ, weight).
INTEGRATION POINTS:
  - multiverse/core.py (bootstrapping & reality proposals)
  - mathematics/quantum_consciousness.py (S_conscious terms)
  - integration/protocol_weaver.py (sanity checks)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Callable
import numpy as np
from .quantum_consciousness import cognitive_potential

Array = np.ndarray

def _safe_norm(x: Array) -> float:
    return float(np.linalg.norm(x))

@dataclass
class ActionTerms:
    """Callable action components: return scalar action value."""
    S_geom: Callable[[Array], float]
    S_field: Callable[[Array], float]
    S_conscious: Callable[[Array], float]

def default_actions(Dg:int=6, Df:int=64, Dc:int=32) -> ActionTerms:
    def S_geom(g: Array) -> float:
        # toy: penalize curvature-like deviations (quadratic form)
        return 0.5 * float(np.dot(g, g))
    def S_field(phi: Array) -> float:
        # toy KG-like discretized energy (gradient + mass)
        grad = np.gradient(phi)
        m2 = 0.05
        return float(0.5*np.mean(grad[0]**2) + 0.5*m2*np.mean(phi**2))
    def S_conscious(psi_c: Array) -> float:
        # potential over norm density
        rho = float(np.vdot(psi_c, psi_c).real)
        return cognitive_potential(rho, alpha=0.08)
    return ActionTerms(S_geom, S_field, S_conscious)

def total_action(acts: ActionTerms, g: Array, phi: Array, psi_c: Array) -> float:
    return acts.S_geom(g) + acts.S_field(phi) + acts.S_conscious(psi_c)

@dataclass
class WavefunctionalSampler:
    Dg: int = 6
    Df: int = 64
    Dc: int = 32
    temp: float = 0.3
    step: float = 0.2
    seed: int = 0
    acts: ActionTerms = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if self.acts is None:
            self.acts = default_actions(self.Dg, self.Df, self.Dc)

    def init_point(self) -> Tuple[Array, Array, Array]:
        g   = self.rng.normal(size=self.Dg)
        phi = self.rng.normal(size=self.Df)
        psi = self.rng.normal(size=self.Dc) + 1j*self.rng.normal(size=self.Dc)
        nrm = np.linalg.norm(psi)
        psi = psi if nrm < 1e-12 else psi / nrm
        return g, phi, psi

    def mh_step(self, g: Array, phi: Array, psi: Array) -> Tuple[Array, Array, Array]:
        g_p   = g   + self.step * self.rng.normal(size=g.size)
        phi_p = phi + self.step * self.rng.normal(size=phi.size)
        psi_p = psi + self.step * (self.rng.normal(size=psi.size) + 1j*self.rng.normal(size=psi.size))
        psi_p = psi_p / (np.linalg.norm(psi_p) + 1e-12)

        S  = total_action(self.acts, g,   phi,   psi)
        Sp = total_action(self.acts, g_p, phi_p, psi_p)
        log_acc = -(Sp - S) / max(1e-6, self.temp)
        if np.log(self.rng.random()) < log_acc:
            return g_p, phi_p, psi_p
        return g, phi, psi

    def sample(self, steps: int = 200) -> Dict[str, Array | float]:
        g, phi, psi = self.init_point()
        for _ in range(steps):
            g, phi, psi = self.mh_step(g, phi, psi)
        S = total_action(self.acts, g, phi, psi)
        weight = np.exp(-S / max(1e-6, self.temp))
        return {"g": g, "phi": phi, "psi": psi, "action": S, "weight": float(weight)}

if __name__ == "__main__":
    wf = WavefunctionalSampler(Dg=4, Df=32, Dc=16, seed=7)
    out = wf.sample(150)
    print({k: (round(v,4) if isinstance(v,float) else (v.shape, float(np.linalg.norm(v)) if k!='phi' else np.var(v)))
           for k,v in out.items()})
