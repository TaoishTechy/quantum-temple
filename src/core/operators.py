import numpy as np
from .state import ResonanceState

def H_stab(state: ResonanceState, g=1.0):
    phases = np.array([n.phase for n in state.nodes])
    mean_phase = np.angle(np.sum(np.exp(1j * phases)))
    for n in state.nodes:
        n.apply_phase(0.01 * (mean_phase - n.phase))

def H_obs(state: ResonanceState, sigma_Q: float = 0.10):
    phases = np.array([n.phase for n in state.nodes])
    coupling = sigma_Q * np.sin(phases - phases.mean())
    for n, c in zip(state.nodes, coupling):
        n.apply_phase(0.05 * c)

def F_retro(state: ResonanceState, tau_steps: int = 16):
    if len(state.psi_delay) >= tau_steps:
        delayed = state.psi_delay[-tau_steps]
        dmean = np.angle(np.sum(np.exp(1j * delayed)))
        for n in state.nodes:
            n.apply_phase(0.005 * (dmean - n.phase))

def apply_Hcrit(state: ResonanceState, B: dict, *, sigma_Q=0.10, tau_steps=16, g=1.0):
    H_stab(state, g)
    H_obs(state, sigma_Q=sigma_Q)
    F_retro(state, tau_steps=tau_steps)
