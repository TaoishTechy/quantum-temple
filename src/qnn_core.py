"""
Quantum Neural Network Core – v1.2
Implements SigilFactory (quantum channels), QuantumSystem (state evolution),
and QNNTrainer (evolutionary optimization). Fully CPTP-safe and scalable.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────
# Quantum SigilFactory
# ──────────────────────────────────────────────────────────────

class Sigil:
    def __init__(self, name, kraus_ops):
        self.name = name
        self.kraus_ops = kraus_ops


class SigilFactory:
    def __init__(self, num_qudits, dimension):
        self.num_qudits = num_qudits
        self.dimension = dimension

    def create_decoherence_sigil(self, name, target_qudit, rate):
        """
        CPTP-safe dephasing channel.
        Kraus ops: { √(1-p)·I, √p·Z }
        """
        p = max(0.0, min(1.0, float(rate)))
        I_d = np.identity(self.dimension, dtype=complex)
        Z_d = np.diag(np.exp(2j * np.pi * np.arange(self.dimension) / self.dimension))

        def lift(op):
            ops = [I_d for _ in range(self.num_qudits)]
            ops[target_qudit] = op
            full = ops[0]
            for k in range(1, self.num_qudits):
                full = np.kron(full, ops[k])
            return full

        K0 = np.sqrt(1 - p) * lift(I_d)
        K1 = np.sqrt(p) * lift(Z_d)
        return Sigil(name, [K0, K1])


# ──────────────────────────────────────────────────────────────
# Quantum System and Hamiltonian
# ──────────────────────────────────────────────────────────────

class QuantumSystem:
    def __init__(self, num_qudits=2, dimension=6):
        self.num_qudits = num_qudits
        self.dimension = dimension
        self.size = dimension ** num_qudits
        self.state_vector = np.zeros(self.size, dtype=complex)
        self.state_vector[0] = 1.0  # |00...0>
        self.hamiltonian = np.zeros((self.size, self.size), dtype=complex)

    def update_model(self, couplings):
        """Apply coupling matrix to Hamiltonian (Hermitian symmetrized)."""
        n = couplings.shape[0]
        if n != self.size:
            raise ValueError("Coupling matrix dimension mismatch")
        self.hamiltonian = 0.5 * (couplings + couplings.conj().T)

    def perceive(self, time_step=0.1):
        """
        Efficient evolution using expm_multiply (sparse propagation).
        Returns density matrix ρ.
        """
        Hs = csr_matrix(self.hamiltonian)
        ψ0 = self.state_vector
        ψ1 = expm_multiply((-1j * Hs * time_step), ψ0)
        norm = np.linalg.norm(ψ1)
        if norm > 1e-12:
            ψ1 /= norm
        self.state_vector = ψ1
        return np.outer(ψ1, ψ1.conj())


# ──────────────────────────────────────────────────────────────
# Trainer – Free Energy Minimization
# ──────────────────────────────────────────────────────────────

class QNNTrainer:
    def __init__(self, system: QuantumSystem):
        self.system = system
        self.cost_history = []

    def _assemble_couplings(self, params):
        n = self.system.hamiltonian.shape[0]
        C = np.zeros((n, n))
        iu = np.triu_indices(n, k=1)
        C[iu] = params
        C += C.T
        return C

    def _free_energy_cost(self, params, data, l2=1e-3):
        C = self._assemble_couplings(params)
        self.system.update_model(C)
        costs = []
        for _, y in data:
            ρ = self.system.perceive()
            f = float(np.real(np.vdot(y, ρ @ y)))
            costs.append(1 - f)
        return np.mean(costs) + l2 * np.sum(C ** 2)

    def train(self, data, epochs=50):
        n = self.system.hamiltonian.shape[0]
        x0 = np.random.normal(0, 0.1, size=(n * (n - 1) // 2,))
        objective = lambda p: self._free_energy_cost(p, data)
        res = minimize(objective, x0, method="COBYLA", options={"maxiter": epochs})
        self.system.update_model(self._assemble_couplings(res.x))
        final = objective(res.x)
        self.cost_history.append(final)
        print(f"✅ Training complete: final cost = {final:.6f}")


# ──────────────────────────────────────────────────────────────
# Example standalone run (for dev testing)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    qs = QuantumSystem(num_qudits=2, dimension=4)
    trainer = QNNTrainer(qs)
    dummy_y = np.random.rand(qs.size) + 1j * np.random.rand(qs.size)
    dummy_y /= np.linalg.norm(dummy_y)
    dataset = [(None, dummy_y)]
    trainer.train(dataset, epochs=10)
