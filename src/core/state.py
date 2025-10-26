from dataclasses import dataclass, field
import numpy as np
from collections import deque

@dataclass
class QuditNode:
    phase: float = 0.0
    amplitude: float = 1.0
    def measure_phase(self) -> float: return self.phase
    def apply_phase(self, dphi: float) -> None: self.phase += dphi

@dataclass
class ResonanceState:
    nodes: list[QuditNode]
    t: float = 0.0
    dt: float = 1e-3
    history_len: int = 4096
    phases_hist: deque = field(default_factory=lambda: deque(maxlen=4096))
    # buffers for retro feed-forward
    psi_delay: deque = field(default_factory=lambda: deque(maxlen=1024))  # Ψ(t-τ)
