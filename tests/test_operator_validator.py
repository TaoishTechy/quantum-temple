# FILE: tests/test_operator_validator.py
import numpy as np
from quantum_temple_multiverse.ops.operator_validator import project_hermitian, nearest_unitary

def test_hermitian_and_unitary_repairs():
    A = np.array([[1,2j],[-1j,0.1]], complex)
    H = project_hermitian(A)
    assert np.allclose(H, H.conj().T)
    U = nearest_unitary(A)
    assert np.allclose(U.conj().T@U, np.eye(2))
