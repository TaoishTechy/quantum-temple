import numpy as np

def phase_lock_value(phases) -> float:
    a = np.asarray(phases)
    return float(np.abs(np.mean(np.exp(1j*a))))

def early_warnings(series) -> dict:
    s = np.asarray(series)
    var = float(np.var(s))
    s = s - s.mean()
    denom = (np.linalg.norm(s[:-1])*np.linalg.norm(s[1:]) + 1e-12)
    ac1 = 0.0 if len(s) < 2 else float(np.correlate(s[:-1], s[1:]) / denom)
    return {"variance": var, "lag1": ac1}

def aesthetic_manifold(N: float, EP: float, E: float) -> float:
    return float(N * EP * E)
