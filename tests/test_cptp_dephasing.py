import numpy as np
from src.qnn_core import SigilFactory

def test_dephasing_cptp_unit_trace():
    # dimension 6, 2 qudits example
    sig = SigilFactory(num_qudits=2, dimension=6).create_decoherence_sigil(
        "T2", target_qudit=0, rate=0.3
    )
    # ∑ K†K = I check (within numeric tolerance)
    acc = None
    for K in sig.kraus_ops:
        term = K.conj().T @ K
        acc = term if acc is None else acc + term
    I = np.eye(acc.shape[0])
    assert np.allclose(acc, I, atol=1e-9)
