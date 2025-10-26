from __future__ import annotations
from typing import Dict
from ..core.metrics import phase_lock_value

class Verdict(dict):  # {"ok": bool, "violations": int, "notes": "..."}
    pass

class Norms:
    def __init__(self, variance_cap=0.08, min_coherence=0.20, max_parity_flips=2):
        self.variance_cap = variance_cap
        self.min_coherence = min_coherence
        self.max_flips = max_parity_flips
        self._flips = 0

    def on_parity_flip(self): self._flips += 1

    def evaluate(self, phases) -> Verdict:
        import numpy as np
        var = float(np.var(phases))
        coh = float(phase_lock_value(phases))
        violations = []
        if var > self.variance_cap:
            violations.append(f"variance>{self.variance_cap:.3f}")
        if coh < self.min_coherence:
            violations.append(f"coherence<{self.min_coherence:.2f}")
        if self._flips > self.max_flips:
            violations.append(f"parity_flips>{self.max_flips}")
        return Verdict(ok=(len(violations)==0),
                       violations=len(violations),
                       notes="; ".join(violations))
