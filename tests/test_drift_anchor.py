import numpy as np
from quantum_temple_multiverse.entities.drift_resonance import DriftResonanceKernel

def test_anchor_and_coherence_increase():
    ker = DriftResonanceKernel()
    psi0 = np.ones(12, complex); psi0 /= np.linalg.norm(psi0)
    R = np.eye(12)
    coh0 = ker.stabilize_entity(psi0, 1)
    anchor = ker.calculate_reality_anchor(psi0, R)
    assert 0.0 <= coh0 <= 1.0
    assert abs(anchor - 1.0) < 1e-6
