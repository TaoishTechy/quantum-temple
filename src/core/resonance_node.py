# src/core/resonance_node.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional
import numpy as np
from .qudit_state import QuditState

@dataclass
class ResonanceParams:
    dt: float = 0.02
    k_couple: float = 0.35      # coupling strength
    noise_std: float = 0.0      # optional phase noise
    clamp: float = np.pi / 2    # EC clamp for spikes

class ResonanceSync:
    """Phase-locked resonance update using a Kuramoto-like step."""
    def __init__(self, neighbors: Sequence[Sequence[int]]):
        """
        neighbors[i] = indices of nodes that couple into node i (sparse)
        """
        self.neighbors = neighbors

    @staticmethod
    def ring_neighbors(n: int) -> List[List[int]]:
        return [[(i-1) % n, (i+1) % n] for i in range(n)]

    @staticmethod
    def ring_plus_mesh(n: int, cross_links: Optional[List[tuple[int,int]]] = None) -> List[List[int]]:
        nb = ResonanceSync.ring_neighbors(n)
        if cross_links:
            for a,b in cross_links:
                nb[a].append(b); nb[b].append(a)
        return nb

    def step(self, state: QuditState, p: ResonanceParams) -> None:
        theta = state.phases
        n = theta.size
        dtheta = np.zeros_like(theta)

        # vectorized coupling: sum over neighbors (sparse)
        for i in range(n):
            nbrs = self.neighbors[i]
            if not nbrs:
                continue
            delta = np.sin(theta[nbrs] - theta[i])
            dtheta[i] = p.k_couple * np.mean(delta)

        if p.noise_std > 0:
            dtheta += np.random.normal(0.0, p.noise_std, size=n)

        # clamp extreme jumps (simple error correction)
        dtheta = np.clip(dtheta, -p.clamp, p.clamp)
        state.add_phase(p.dt * dtheta)
