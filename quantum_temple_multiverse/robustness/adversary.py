"""
FILE: quantum_temple_multiverse/robustness/adversary.py
PURPOSE: Targeted perturbations for states/operators; Byzantine node tests.
MATHEMATICAL CORE:
  - FGSM-like gradient step on a scalar objective J(ψ)
  - Byzantine: random malicious phase flips and norm skews on subset p
INTEGRATION POINTS: entities.drift_resonance, integration.emergent_properties
"""
from __future__ import annotations
import numpy as np

def fgsm_state_attack(psi: np.ndarray, grad_fn, eps: float=0.01) -> np.ndarray:
    # grad_fn returns ∂J/∂ψ*
    g = grad_fn(psi)
    adv = psi + eps * g / (np.linalg.norm(g) + 1e-12)
    return adv / (np.linalg.norm(adv) + 1e-12)

def byzantine_phases(psi_list: list[np.ndarray], fraction: float=0.33, seed:int=0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(psi_list); k = max(1, int(fraction*n))
    bad = set(rng.choice(n, size=k, replace=False))
    out=[]
    for i,psi in enumerate(psi_list):
        if i in bad:
            phase = np.exp(1j * rng.uniform(-np.pi, np.pi))
            x = psi*phase + 0.1*rng.normal(size=psi.size)
            x = x/(np.linalg.norm(x)+1e-12)
            out.append(x)
        else:
            out.append(psi.copy())
    return out

if __name__ == "__main__":
    psi = np.ones(6,complex)/np.sqrt(6)
    gfn = lambda p: -p  # maximize norm on |0> direction proxy
    adv = fgsm_state_attack(psi, gfn, 0.2)
    print("||adv||≈", round(np.linalg.norm(adv),4))
