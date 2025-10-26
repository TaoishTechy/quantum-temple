# src/structures/quantum_heap.py
from __future__ import annotations
import heapq, numpy as np
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class HeapItem:
    priority: float
    item_id: int = field(compare=False)
    payload: Any = field(compare=False)

class QuantumHeap:
    """
    Probabilistic heap for state snapshots or operations.
    - push(state, score): lower score => higher priority
    - pop() returns a sample biased by softmax over inverse priority
    """
    def __init__(self, temperature: float = 0.1, seed: int = 0):
        self._heap = []
        self._rng = np.random.default_rng(seed)
        self._counter = 0
        self.temperature = max(1e-6, float(temperature))

    def push(self, payload: Any, score: float) -> None:
        heapq.heappush(self._heap, HeapItem(priority=float(score), item_id=self._counter, payload=payload))
        self._counter += 1

    def _weights(self):
        if not self._heap:
            return np.array([])
        pr = np.array([h.priority for h in self._heap], dtype=float)
        inv = -pr / self.temperature
        inv -= inv.max()  # stabilize
        w = np.exp(inv)
        return w / w.sum()

    def sample(self):
        if not self._heap:
            return None
        w = self._weights()
        idx = self._rng.choice(len(self._heap), p=w)
        return self._heap[idx].payload

    def pop(self):
        """Deterministic pop: lowest priority (best score) first."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap).payload
