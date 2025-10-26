import numpy as np
from ..core.state import ResonanceState
from ..core.operators import apply_Hcrit
from ..core.control import SigmaQPID, PiLock, rla_anchor
from ..core.metrics import phase_lock_value, early_warnings
from ..core.ledger import Ledger

class TempleEngine:
    def __init__(self, state: ResonanceState, alpha=0.12, init_sigma=0.09, var_target=0.05):
        self.state = state
        self.pid = SigmaQPID(target_var=var_target, init_sigma=init_sigma)
        self.pilock = PiLock(budget=2)
        self.anchor = rla_anchor(ci_target=0.98, alpha=alpha)
        self.ledger = Ledger()

    def step(self):
        phases = np.array([n.phase for n in self.state.nodes])
        ew = early_warnings(phases)
        sigma = self.pid.step(ew["variance"], dt=self.state.dt)

        self.state.psi_delay.append(phases.copy())
        apply_Hcrit(self.state, B={"gain": 1.0}, sigma_Q=sigma, tau_steps=16, g=1.0)

        coh = phase_lock_value([n.phase for n in self.state.nodes])
        flipped = self.pilock.try_flip(coherence=coh)

        self.state.t += self.state.dt
        self.state.phases_hist.append(coh)

        self.ledger.append("tick", {
            "t": self.state.t,
            "sigma_Q": sigma,
            "coherence": coh,
            "variance": ew["variance"],
            "lag1": ew["lag1"],
            "parity_flip": flipped
        })
