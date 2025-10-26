"""
FILE: quantum_temple_multiverse/ops/operator_validator.py
PURPOSE: Validate/repair Hermiticity, Unitarity; validate CPTP via Choi positivity.
MATHEMATICAL CORE:
  - Hermitian projection H ← (H+H†)/2
  - Nearest unitary U ← polar(H) via SVD
  - CPTP: Choi(Φ) ≥ 0 and Tr-preserving
INTEGRATION POINTS: integration.protocol_weaver, civilizations.narrative_engine
"""
from __future__ import annotations
import numpy as np

def project_hermitian(A: np.ndarray) -> np.ndarray:
    return 0.5*(A + A.conj().T)

def nearest_unitary(A: np.ndarray) -> np.ndarray:
    U,_,Vh = np.linalg.svd(A)
    return U@Vh

def choi_from_kraus(kraus:list[np.ndarray]) -> np.ndarray:
    # Choi = sum_k (K_k ⊗ I) |Ω⟩⟨Ω| (K_k† ⊗ I), with |Ω⟩ = vec(I)
    K0 = kraus[0]
    d = K0.shape[0]
    choi = np.zeros((d*d, d*d), complex)
    vecI = np.eye(d).reshape(d*d,1)
    for K in kraus:
        A = np.kron(K, np.eye(d)) @ vecI
        choi += A @ A.conj().T
    return choi

def is_cptp(kraus:list[np.ndarray], tol:float=1e-6) -> bool:
    K0 = kraus[0]; d = K0.shape[0]
    choi = choi_from_kraus(kraus)
    eig = np.linalg.eigvalsh(choi)
    # trace preserving: sum_k K_k†K_k = I
    tp = np.zeros((d,d), complex)
    for K in kraus: tp += K.conj().T @ K
    return (eig.min() >= -tol) and (np.linalg.norm(tp - np.eye(d)) < 1e-6)

if __name__ == "__main__":
    A = np.array([[1,2j],[ -2j,3]], complex)
    H = project_hermitian(A)
    U = nearest_unitary(A)
    print("hermitian?", np.allclose(H, H.conj().T))
    print("unitary?", np.allclose(U.conj().T@U, np.eye(2)))
