"""
FILE: quantum_temple_multiverse/expansion/driftwave.py
PURPOSE: Generate and score driftwave propagation (nonlinear scalar field).
MATHEMATICAL CORE:
  □Φ + m^2 Φ + λ Φ^3 = J   (1+1D toy discretization, leapfrog)
  FRW metric sample: ds² = -dt² + a(t)² dx²  => we return a(t)=exp(Ht)
INTEGRATION POINTS: multiverse.core, mathematics.multiversal_metrics
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class DriftwaveExpansionCapsule:
    dx: float = 0.25
    dt: float = 0.05
    m2: float = 0.1
    lam: float = 0.05
    H: float = 0.01  # simple exponential scale factor parameter

    def generate_driftwave(self, source_profile: np.ndarray, steps: int = 50, J: float = 0.0):
        """
        Leapfrog for 1D field; returns Φ(t_final) and energy-like norm.
        """
        N = source_profile.size
        phi = source_profile.astype(float).copy()
        phi_old = phi.copy()  # start at rest
        for _ in range(steps):
            lap = (np.roll(phi,1)-2*phi+np.roll(phi,-1)) / (self.dx**2)
            # KG-like update with cubic nonlinearity + source
            phi_new = (2*phi - phi_old +
                       (self.dt**2) * (lap - self.m2*phi - self.lam*(phi**3) + J))
            phi_old, phi = phi, phi_new
        energy = float(np.mean(0.5*(phi**2) + 0.5*(np.gradient(phi,self.dx)**2) + 0.25*self.lam*(phi**4)))
        return phi, energy

    def calculate_expansion_metric(self, t: float) -> float:
        # a(t) = exp(H t)
        return float(np.exp(self.H * t))

if __name__ == "__main__":
    cap = DriftwaveExpansionCapsule()
    src = np.exp(-((np.linspace(-5,5,128))**2))
    phi, E = cap.generate_driftwave(src, 30, J=0.01)
    print("driftwave_E≈", round(E,6), "a(t=10)≈", round(cap.calculate_expansion_metric(10.0),4))
