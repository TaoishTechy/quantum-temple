"""
FILE: quantum_temple_multiverse/metrics/decoherence_tomography.py
PURPOSE: Identify dominant decoherence (dephasing vs amplitude damping) and estimate rates.
MATHEMATICAL CORE:
  - Fit single-qudit Lindblad surrogates by least squares over observed snapshots.
  - Channels: PhaseDamp(λ), AmpDamp(γ) on basis {|k⟩}, treat pairwise subspaces.
INTEGRATION POINTS: entities.drift_resonance, mathematics.quantum_consciousness
"""
from __future__ import annotations
import numpy as np

def phase_damp(rho: np.ndarray, lam: float) -> np.ndarray:
    r = rho.copy()
    i,j = np.indices(r.shape)
    mask = (i!=j)
    r[mask] *= (1.0 - lam)
    return r

def amp_damp_2lvl(rho2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Two-level amplitude damping (|1>→|0>) Kraus:
      K0 = [[1,0],[0, sqrt(1-γ)]], K1 = [[0, sqrt(γ)],[0,0]]
    """
    g = float(np.clip(gamma, 0.0, 1.0))
    K0 = np.array([[1.0,0.0],[0.0,np.sqrt(1.0-g)]], complex)
    K1 = np.array([[0.0,np.sqrt(g)],[0.0,0.0]], complex)
    return K0@rho2@K0.conj().T + K1@rho2@K1.conj().T

def fit_phase_damp(rho_t0: np.ndarray, rho_t1: np.ndarray) -> float:
    # minimize Frobenius || phase_damp(rho_t0, λ) - rho_t1 ||
    best,best_err = 0.0, 1e99
    for lam in np.linspace(0.0,1.0,101):
        err = np.linalg.norm(phase_damp(rho_t0, lam) - rho_t1)
        if err < best_err: best, best_err = lam, err
    return float(best)

def fit_amp_damp_pair(rho_t0: np.ndarray, rho_t1: np.ndarray, k0:int, k1:int) -> float:
    # extract 2x2 subspace density and fit γ
    idx = np.ix_([k0,k1],[k0,k1])
    r0, r1 = rho_t0[idx], rho_t1[idx]
    best,best_err = 0.0, 1e99
    for g in np.linspace(0.0,1.0,101):
        err = np.linalg.norm(amp_damp_2lvl(r0,g) - r1)
        if err < best_err: best,best_err = g, err
    return float(best)

def characterize_decoherence(rho_t0: np.ndarray, rho_t1: np.ndarray) -> dict:
    lam = fit_phase_damp(rho_t0, rho_t1)
    # scan a few pairs (00-11-22...) for amplitude damping tendencies
    d = rho_t0.shape[0]
    pairs = [(i,(i+1)%d) for i in range(d)]
    gammas = [fit_amp_damp_pair(rho_t0, rho_t1, a,b) for a,b in pairs]
    return {"phase_damp_lambda": lam, "amp_damp_gamma_mean": float(np.mean(gammas))}

if __name__ == "__main__":
    # synthetic check
    d=5
    psi = np.ones(d,complex)/np.sqrt(d)
    rho0 = np.outer(psi, psi.conj())
    rho1 = phase_damp(rho0, 0.2)
    print(characterize_decoherence(rho0, rho1))
