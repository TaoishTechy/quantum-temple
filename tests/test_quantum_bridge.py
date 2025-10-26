import numpy as np
from quantum_temple_multiverse.integration.quantum_bridge import QuditCognitionBridge

def test_qudit_cognition_roundtrip():
    br = QuditCognitionBridge(d=5, n=3)
    c = (np.ones(20) + 1j) / np.sqrt(40)
    psi = br.to_qudit(c)
    c2 = br.to_cognition(psi, K=20)
    assert psi.size == 125
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-6
    assert np.linalg.norm(c2 - c) < 1e-8
