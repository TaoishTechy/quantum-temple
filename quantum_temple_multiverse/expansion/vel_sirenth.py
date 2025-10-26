"""
FILE: quantum_temple_multiverse/expansion/vel_vohr.py
PURPOSE: Nullspace Operations — safe projections & null directions.
MATHEMATICAL CORE: Nullspace via SVD; projection P = I - A^+ A
INTEGRATION POINTS: entities.drift_resonance, multiverse.core
"""
from __future__ import annotations
import numpy as np

class VelVohrNullspace:
    def nullspace(self, A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        U, s, Vh = np.linalg.svd(A)
        null_mask = (s <= eps)
        if not null_mask.any():
            # smallest singular vector as "approximate" null
            vec = Vh[-1,:]
            return vec / (np.linalg.norm(vec) + 1e-12)
        V = Vh.T
        N = V[:, null_mask]
        # return first null vector normalized
        vec = N[:,0]
        return vec / (np.linalg.norm(vec) + 1e-12)

    def projector(self, A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        # P = I - A^+ A
        Ap = np.linalg.pinv(A, rcond=eps)
        I = np.eye(A.shape[1])
        return I - Ap @ A

if __name__ == "__main__":
    nv = VelVohrNullspace()
    A = np.array([[1,0,0],[0,1,0]], float)
    v = nv.nullspace(A)
    P = nv.projector(A)
    print("null_vec_norm≈", round(np.linalg.norm(v),4), "proj_rank≈", int(np.linalg.matrix_rank(P)))
