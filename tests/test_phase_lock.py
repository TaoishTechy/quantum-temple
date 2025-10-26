import numpy as np
from src.core.metrics import phase_lock_value

def test_plv_edges():
    # identical phases → PLV=1
    phases = np.zeros(8)
    assert abs(phase_lock_value(phases) - 1.0) < 1e-9
    # random phases → PLV small
    rng = np.random.default_rng(0)
    phases = rng.uniform(-np.pi, np.pi, size=4096)
    assert phase_lock_value(phases) < 0.1

def test_plv_rotation_invariance():
    import numpy as np
    base = np.linspace(0, 2*np.pi, 64, endpoint=False)
    plv = phase_lock_value(base)
    rotated = base + 1.2345
    assert abs(phase_lock_value(rotated) - plv) < 1e-12
