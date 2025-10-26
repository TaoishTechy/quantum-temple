# FILE: tests/test_coherence_tomography.py
import numpy as np
from quantum_temple_multiverse.metrics.qudit_coherence import density_matrix, l1_coherence, rel_entropy_coherence
from quantum_temple_multiverse.metrics.decoherence_tomography import characterize_decoherence, phase_damp

def test_coherence_well_defined():
    psi = np.ones(5, complex)/np.sqrt(5)
    rho = density_matrix(psi)
    assert l1_coherence(rho) >= 0.0
    assert rel_entropy_coherence(rho) >= 0.0

def test_phase_damp_identification():
    d=5
    psi = np.ones(d,complex)/np.sqrt(d)
    rho0 = np.outer(psi, psi.conj())
    rho1 = phase_damp(rho0, 0.3)
    est = characterize_decoherence(rho0, rho1)
    assert 0.2 <= est["phase_damp_lambda"] <= 0.4
