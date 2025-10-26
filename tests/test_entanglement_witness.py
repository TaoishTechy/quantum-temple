# FILE: tests/test_entanglement_witness.py
import numpy as np
from quantum_temple_multiverse.metrics.entanglement_witness import is_entangled_bipartite

def test_ppt_ccnr_detects_simple_maxent():
    d=3
    psi = np.zeros(d*d,complex); psi[0]=psi[4]=psi[8]=1.0; psi/=np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    res = is_entangled_bipartite(rho, d, d)
    assert res["entangled"] is True
