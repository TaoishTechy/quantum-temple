import numpy as np

def phase_lock_value(phases: np.ndarray) -> float:
    return np.abs(np.mean(np.exp(1j*phases)))

def criticality_index(lambda_dom_now: float, lambda_base: float) -> float:
    return 1.0 - abs(lambda_dom_now)/max(1e-12, abs(lambda_base))

def early_warnings(series: np.ndarray) -> dict:
    # crude: variance + lag-1 autocorrelation
    var = float(np.var(series))
    s = series - series.mean()
    ac1 = float(np.correlate(s[:-1], s[1:]) / (np.linalg.norm(s[:-1])*np.linalg.norm(s[1:]) + 1e-12))
    return {"variance": var, "lag1": ac1}

def aesthetic_manifold(N: float, EP: float, E: float) -> float:
    return N * EP * E
