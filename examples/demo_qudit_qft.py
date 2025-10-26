# examples/demo_qudit_qft.py
import numpy as np
from src.quantum.qudit_space import QuantumRegister
from src.operators.qudit_gates import DFT, CADD
from src.algorithms.qudit_qft import qft_register, iqft_register
from src.measurement.measurement import probabilities

d, n = 5, 2
qr = QuantumRegister(n_qudits=n, d=d, seed=42)
# put |11> by shifting twice on both qudits with X gates
from src.operators.qudit_gates import X
qr.apply_gate(X(d) @ X(d), target=0)  # quick composite example (small n)
qr.apply_gate(X(d) @ X(d), target=1)  # doesn't actually entangle; demo only

psi_f = qft_register(qr.state, d=d, n=n)
psi_b = iqft_register(psi_f, d=d, n=n)
print("roundtrip error:", np.linalg.norm(psi_b - qr.state))
print("out probs (first 10):", probabilities(psi_f)[:10])
