"""
FILE: quantum_temple_multiverse/sim/causality.py
PURPOSE: Enforce causal update with discrete delay τ (ring topologies).
MATHEMATICAL CORE:
  ψ_i(t+Δt) depends only on {ψ_j(t), ψ_j(t-τ), …}; ring buffer implements delay lines.
INTEGRATION POINTS: multiverse.core, civilizations.symbolic_observatory
"""
from __future__ import annotations
import numpy as np
from collections import deque

class DelayLine:
    def __init__(self, length:int):
        self.buf = deque([None]*length, maxlen=length)
    def push(self, x): self.buf.append(x)
    def peek_delay(self, k:int):  # k=1 => one-step delay
        if k<=0 or k>len(self.buf): raise ValueError("bad delay")
        return list(self.buf)[-k]

class CausalRing:
    def __init__(self, N:int, delay:int):
        self.N=N; self.delay=delay
        self.lines=[DelayLine(delay) for _ in range(N)]
    def step(self, update_fn):
        """
        update_fn(i, now_state, delayed_neighbor_state) -> new_state
        All updates computed from previous buffers; then committed.
        """
        prev=[ln.peek_delay(1) for ln in self.lines]
        new=[]
        for i in range(self.N):
            left = self.lines[(i-1)%self.N].peek_delay(self.delay)
            new.append(update_fn(i, prev[i], left))
        for ln,x in zip(self.lines, new): ln.push(x)

if __name__ == "__main__":
    ring = CausalRing(3, delay=2)
    for ln in ring.lines: ln.push(0.0); ln.push(1.0)
    def f(i, x, l): return 0.5*(x + l)
    ring.step(f)
    print("ok causal; head:", [ln.peek_delay(1) for ln in ring.lines])
