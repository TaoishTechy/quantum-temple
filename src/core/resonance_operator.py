from sympy import symbols, exp, I
from numpy import mean, angle, exp as npexp, array

def resonance_operator(qudits, eta=0.01, phase_lock=True):
    """Riemann-zeta symbolic phase operator"""
    zeta_shift = [exp(I * eta * i) for i, _ in enumerate(qudits)]
    qudits = [q * z for q, z in zip(qudits, zeta_shift)]
    if phase_lock:
        from .phase_lock_step import phase_lock_step
        qudits = phase_lock_step(qudits)
    return qudits
